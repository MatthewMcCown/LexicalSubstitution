# LexicalSubstitution
Lexical substitution for words in context

The goal is to find substitutions for target words in context that preserve the meaning of the sentences. This approach uses both WordNet and pre-trained Word2Vec word embeddings.

For example, given the sentence: "The project budget was ###tight###."
A plausible substitution is: "The project budget was ###small###."
Whereas the following substitution would be illogical: "The project budget was ###scarce###."
