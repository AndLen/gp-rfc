# gp-rfc

GPRFC entry point is tests.featureCreate.FeatureCreateEuroGP18
GPMVRFC entry point is tests.featureCreate.GPRFCSimMV

Both use JUnit via tests.Tests to run on a range of different datasets automatically, e.g. a_irisTest()

datasets are in the datasets/ folder and use a variation on CSV format. You may want to adapt this...

e.g. iris.data:
classLast,4,3,comma
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,1.4,0.2,Iris-setosa
....

classLast: the class label is last (vs classFirst);
4: 4 features;
3: 3 classes;
comma: comma-separated file (vs space for space-separated or tab for tab-separated).
