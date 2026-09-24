SYSTEM_PROMPT = (
    "És um argumentista profissional de comédia e sátira portuguesa.\n"
    "Escreves exclusivamente em Português Europeu (PT-PT), usando o vocabulário, "
    "a sintaxe e as expressões idiomáticas correntes em Portugal.\n"
    "O teu humor é observacional, irónico e subversivo: ancoras as piadas na "
    "realidade social, política e quotidiana portuguesa e escalas o absurdo a "
    "partir de premissas reconhecíveis.\n"
    "Nunca explicas a piada depois de a fazeres e "
    "preferes o risco cómico à segurança de um texto genérico.\n"
    "Se te forem dados sketches de referência, usa-os apenas como modelo de "
    "ritmo, cadência e registo, nunca reaproveites as suas falas ou premissas."
)

MACRO_INSTRUCTION = (
    "Escreve um sketch de comédia original em Português de Portugal a partir do "
    "tema e premissa indicados no fim.\n\n"
    "Antes de escreveres, planeia o arco cómico completo no teu raciocínio interno "
    "seguindo a estrutura: ELENCO, ABORDAGEM REJEITADA, ARCO CÓMICO (com expectativa, "
    "violação e lógica interna para cada piada) e ESCALADA.\n\n"
    "IMPORTANTE: o plano é apenas para teu uso interno. A tua resposta final deve "
    "conter APENAS o guião em falas, sem plano, sem títulos de secção, sem ELENCO, "
    "sem comentários.\n\nRegras de escrita:\n"
    "1. Escreve o guião em falas. Cada fala ocupa uma linha própria, precedida "
    "pelo nome da personagem em maiúsculas entre parênteses retos: [NOME]: fala.\n"
    "2. Podes acrescentar didascálias breves em linha própria, entre parênteses "
    "retos e sem dois pontos, no máximo uma por cada seis falas.\n"
    "3. Fixa as personagens no início e mantém-nas até ao fim.\n"
    "4. Constrói uma escalada: cada piada deve subir a aposta da anterior e "
    "terminar na punchline mais forte.\n"
    "5. Usa referências culturais portuguesas concretas em vez de genéricas.\n"
    "6. Extensão alvo: 500 a 900 palavras.\n\n"
    "Tema e premissa:"
)

NEWSPAPER_INSTRUCTION = (
    "Escreve uma crónica satírica original em Português de Portugal a partir do "
    "tema indicado no fim.\n\n"
    "Antes de escreveres, planeia o arco da crónica no teu raciocínio interno: o "
    "ângulo crítico, a tese absurda que vais defender com aparente seriedade, e "
    "como a escalada de argumentos conduz ao remate final.\n\n"
    "IMPORTANTE: o plano é apenas para teu uso interno. A tua resposta final deve "
    "conter APENAS a crónica, sem plano, sem títulos de secção, sem comentários.\n\n"
    "Regras de escrita:\n"
    "1. Começa com um título satírico numa linha própria.\n"
    "2. Escreve em prosa corrida, em parágrafos. NÃO uses diálogo, nomes de "
    "personagens entre parênteses retos, nem didascálias.\n"
    "3. Escreve na primeira pessoa, como cronista, com um tom de aparente "
    "seriedade que torna o absurdo mais evidente.\n"
    "4. Constrói uma escalada: cada argumento deve ser mais absurdo do que o "
    "anterior, mantendo sempre a lógica interna.\n"
    "5. Não termines com moral nem conclusão explicativa; mantém a ironia até à "
    "última frase.\n"
    "6. Usa referências culturais portuguesas concretas em vez de genéricas.\n"
    "7. Extensão alvo: 500 a 800 palavras.\n\n"
    "Tema:"
)

MONOLOGUE_INSTRUCTION = (
    "Escreve um monólogo humorístico de televisão, estilo late-night, em Português "
    "de Portugal a partir do tema indicado no fim.\n\n"
    "Antes de escreveres, planeia o arco cómico no teu raciocínio interno: como "
    "abres, que observações encadeias, como escalam entre si, e qual é o remate "
    "final.\n\n"
    "IMPORTANTE: o plano é apenas para teu uso interno. A tua resposta final deve "
    "conter APENAS o monólogo, sem plano, sem títulos de secção, sem comentários.\n\n"
    "Regras de escrita:\n"
    "1. Uma única voz, a do apresentador, dirigindo-se directamente ao público. "
    "NÃO uses nomes de personagens entre parênteses retos nem diálogo.\n"
    "2. Escreve em parágrafos curtos, um por bloco de piada, para marcar o ritmo.\n"
    "3. Podes usar marcadores de interacção com a plateia em linha própria, entre "
    "parênteses retos, no máximo três em todo o monólogo: [Pausa para risos].\n"
    "4. Constrói uma escalada: cada observação deve subir a aposta da anterior e "
    "terminar no remate mais forte.\n"
    "5. Usa referências culturais portuguesas concretas em vez de genéricas.\n"
    "6. Extensão alvo: 500 a 800 palavras.\n\n"
    "Tema:"
)

INSTRUCTIONS = {
    "sketch": MACRO_INSTRUCTION,
    "newspaper": NEWSPAPER_INSTRUCTION,
    "tv_show": MONOLOGUE_INSTRUCTION,
}


def get_instruction(format_type: str) -> str:
    return INSTRUCTIONS.get(format_type, MACRO_INSTRUCTION)
