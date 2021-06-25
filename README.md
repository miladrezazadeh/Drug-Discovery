# Drug-Discovery

This is an assignment provided during Neural Network Programming course offered by Advanced Research Computing (Scinet) at the University of Toronto.

Assignment description:

An active area of research is the use of neural networks to discover new drugs. In principle we'd like to be able to predict the characteristics of a potential drug without actually measuring the chemical's properties. This would allow much-more rapid development of possible drug candidates, and potentially the ability to discover chemicals with custom properties.

Suppose we want to design a neural network which would take SMILES-format strings as inputs. The natural way to process these inputs would be to build a recurrent neural network, using LSTMs, which would process the string as a "sentence". Once processed, the data could be then used to predict the properties of the chemical.

Let us consider a collection of chemicals with annotated nanomolar activities (IC/EC/AC50), which were downloaded from ChEMBL(https://www.ebi.ac.uk/chembl/), a database of bioactive molecules. The dataset we will consider can be found https://support.scinet.utoronto.ca/education/get.php/chembl.csv.gz. For this assignment, we will be interested in predicting each chemical's value of the "AlogP" column, which is a measure of molecular hydrophobicity (lipophilicity).
