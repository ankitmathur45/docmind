# Hand-crafted test set for RAG evaluation
# Each question has:
#   - question: the query
#   - relevant_chunks: list of chunk texts that should be retrieved
#   - relevance_scores: graded relevance (2=highly relevant, 1=partial, 0=not)
#
# HOW TO BUILD YOUR OWN TEST SET:
# 1. Run your RAG pipeline on each question
# 2. Look at the retrieved chunks
# 3. Mark which ones are actually relevant
# 4. Assign graded scores based on how directly they answer the question

# Replace these with questions and relevant chunks from YOUR actual PDF
TEST_SET = [
    {
        "question": "What is data engineering?",
        "relevant_chunks": [
            "Data engineering is a set of operations aimed at creating interfaces and mechanisms "
            "for the flow and access of information. It takes dedicated specialists—data engineers— "
            "to maintain data so that it",
        ],
        "relevance_scores": {
            "Data engineering is a set of operations aimed at creating interfaces and mechanisms "
            "for the flow and access of information. It takes dedicated specialists—data engineers— "
            "to maintain data so that it": 2,
            "the shadows and now sharing the stage with data science. Data engineering is one of "
            "the hottest fields in data and technology, and for a good reason. It builds the foundation "
            "for data science and an": 1,
            "and processes that take in raw data and produce high -quality, consistent information "
            "that supports downstream use cases, such as analysis and machine learning. Data "
            "engineering is the intersec tion": 1,
        },
    },
    {
        "question": "Explain the key concept of monolith vs modular?",
        "relevant_chunks": [
            "architecture space. Monolithic systems are self -contained, often performing multiple "
            "functions under a single system. The monolith camp favors the simplicity of having "
            "everything in one place. It's",
        ],
        "relevance_scores": {
            "architecture space. Monolithic systems are self -contained, often performing multiple "
            "functions under a single system. The monolith camp favors the simplicity of having "
            "everything in one place. It's": 2,
            "Monolith Versus Modular                                                                                          139 "
            "Monolith                                                                          ": 1,
        },
    },
    {
        "question": "What are principles of good architecture?",
        "relevant_chunks": [
            "Principle 2: Plan for Failure                                                                                      79 "
            "Principle 3: Architect for Scalability                                           ",
        ],
        "relevance_scores": {
            "Principle 2: Plan for Failure                                                                                      79 "
            "Principle 3: Architect for Scalability                                           ": 2,
            "aim to make significant decisions that will lead to good architecture at a basic level. "
            "What do we mean by \"good\" data architecture? To paraphrase an old cliche, you know "
            "good when you see it. Good ": 1,
        },
    },
]