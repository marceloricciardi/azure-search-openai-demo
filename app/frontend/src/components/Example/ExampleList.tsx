import { Example } from "./Example";

import styles from "./Example.module.css";

export type ExampleModel = {
    text: string;
    value: string;
};

const EXAMPLES: ExampleModel[] = [
    {
        text: "O empregado pode recusar-se a trabalhar horas extras?",
        value: "O empregado pode recusar-se a trabalhar horas extras?"
    },
    {
        text: "Qual o período considerado noturno, perante a legislação trabalhista?",
        value: "Qual o período considerado noturno, perante a legislação trabalhista?"
    },
    {
        text: "Qual o valor do acréscimo à remuneração do trabalhador urbano, que realiza tarefa no período noturno?",
        value: "Qual o valor do acréscimo à remuneração do trabalhador urbano, que realiza tarefa no período noturno?"
    }
];

interface Props {
    onExampleClicked: (value: string) => void;
}

export const ExampleList = ({ onExampleClicked }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {EXAMPLES.map((x, i) => (
                <li key={i}>
                    <Example text={x.text} value={x.value} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
