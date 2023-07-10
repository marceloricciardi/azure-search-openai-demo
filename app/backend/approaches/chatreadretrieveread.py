from typing import Any, Sequence

import openai
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from approaches.approach import Approach
from text import nonewlines

class ChatReadRetrieveReadApproach(Approach):
    """
    Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """

    prompt_prefix = """<|im_start|>system
O assistente auxilia os funcionários da empresa com dúvidas sobre as leis trabalhistas e penalidades cabíveis. Seja breve em suas respostas.
Responda APENAS com os fatos listados na lista de fontes abaixo. Se não houver informações suficientes abaixo, diga que não sabe. Não gere respostas que não usem as fontes abaixo. Se fazer uma pergunta esclarecedora ao usuário ajudar, faça a pergunta.
Para obter informações tabulares, retorne-as como uma tabela html. Não retorne o formato de remarcação.
Cada fonte tem um nome seguido de dois pontos e as informaçõeses reais, sempre inclua o nome da fonte para cada fato que você usar na resposta. Use colchetes para referenciar a fonte, por exemplo [info1.txt]. Não combine fontes, liste cada fonte separadamente, por exemplo [info1.txt][info2.pdf].
{follow_up_questions_prompt}
{injected_prompt}
Sources:
{sources}
<|im_end|>
{chat_history}
"""

    follow_up_questions_prompt_content = """Gere três breves perguntas de acompanhamento que o usuário provavelmente faria a seguir sobre as leis trabalhistas (Consolidação das Leis do Trabalho - CLT) e manual do funcionário.
     Use colchetes angulares duplos para fazer referência às perguntas, por exemplo <<Existem exclusões de penalidades?>>.
     Tente não repetir perguntas que já foram feitas.
     Gere apenas perguntas e não gere nenhum texto antes ou depois das perguntas, como 'Próximas perguntas'"""

    query_prompt_template = """Segue abaixo um histórico da conversa até o momento, e uma nova pergunta feita pelo usuário que precisa ser respondida através de uma busca em uma base de conhecimento sobre Consolidação das Leis do Trabalho (CLT) e o manual do empregado.
     Gere uma consulta de pesquisa com base na conversa e na nova pergunta.
     Não inclua nomes de arquivos de origem citados e nomes de documentos, por exemplo, info.txt ou doc.pdf nos termos de consulta de pesquisa.
     Não inclua nenhum texto dentro de [] ou <<>> nos termos da consulta de pesquisa.
     Pesquise no idioma da entrada original da pergunta, nunca tente traduzi-la para o inglês.

Chat History:
{chat_history}

Question:
{question}

Search query:
"""

    def __init__(self, search_client: SearchClient, chatgpt_deployment: str, gpt_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.gpt_deployment = gpt_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def run(self, history: Sequence[dict[str, str]], overrides: dict[str, Any]) -> Any:
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        prompt = self.query_prompt_template.format(chat_history=self.get_chat_history_as_text(history, include_last_turn=False), question=history[-1]["user"])
        completion = openai.Completion.create(
            engine=self.gpt_deployment, 
            prompt=prompt, 
            temperature=0.0, 
            max_tokens=32, 
            n=1, 
            stop=["\n"])
        q = completion.choices[0].text

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query
        if overrides.get("semantic_ranker"):
            r = self.search_client.search(q, 
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC, 
                                          query_language="en-us", 
                                          query_speller="lexicon", 
                                          semantic_configuration_name="default", 
                                          top=top, 
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None)
        else:
            r = self.search_client.search(q, filter=filter, top=top)
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) for doc in r]
        content = "\n".join(results)

        follow_up_questions_prompt = self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""
        
        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_template")
        if prompt_override is None:
            prompt = self.prompt_prefix.format(injected_prompt="", sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
        elif prompt_override.startswith(">>>"):
            prompt = self.prompt_prefix.format(injected_prompt=prompt_override[3:] + "\n", sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
        else:
            prompt = prompt_override.format(sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)

        messages = self.get_messages_from_prompt(prompt)

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history
        chatCompletion = openai.ChatCompletion.create(
            deployment_id=self.chatgpt_deployment,
            model="gpt-3.5-turbo",
            messages=messages, 
            temperature=overrides.get("temperature") or 0.7, 
            max_tokens=1024, 
            n=1, 
            stop=["<|im_end|>", "<|im_start|>"])
        
        chatContent = chatCompletion.choices[0].message.content

        return {"data_points": results, "answer": chatContent, "thoughts": f"Searched for:<br>{q}<br><br>Prompt:<br>" + prompt.replace('\n', '<br>')}
    
    def get_chat_history_as_text(self, history: Sequence[dict[str, str]], include_last_turn: bool=True, approx_max_tokens: int=1000) -> str:
        history_text = ""
        for h in reversed(history if include_last_turn else history[:-1]):
            history_text = """<|im_start|>user""" + "\n" + h["user"] + "\n" + """<|im_end|>""" + "\n" + """<|im_start|>assistant""" + "\n" + (h.get("bot", "") + """<|im_end|>""" if h.get("bot") else "") + "\n" + history_text
            if len(history_text) > approx_max_tokens*4:
                break    
        return history_text
    
    # Generate messages needed for chat Completion api
    from typing import List

    def get_messages_from_prompt(self, prompt: str) -> List[dict[str, str]]:
        messages = []
        for line in prompt.splitlines():
            if line.startswith("<|im_start|>"):
                index = "<|im_start|>".__len__()
                role = line[index:]
            elif line.startswith("<|im_end|>"):
                continue
            else:
                messages.append({"role": role, "content": line})
        return messages
