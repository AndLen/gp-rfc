package featureCreate;

import clustering.CentroidCluster;
import de.lmu.ifi.dbs.elki.data.Cluster;
import de.lmu.ifi.dbs.elki.data.Clustering;
import de.lmu.ifi.dbs.elki.data.model.Model;
import de.lmu.ifi.dbs.elki.database.ids.ArrayModifiableDBIDs;
import de.lmu.ifi.dbs.elki.database.ids.DBIDUtil;
import de.lmu.ifi.dbs.elki.distance.similarityfunction.cluster.ClusteringFowlkesMallowsSimilarityFunction;
import other.PerformanceEvaluation;
import other.Util;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.RandomForest;
import weka.clusterers.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.stream.Stream;

import static weka.clusterers.SimpleKMeans.TAGS_SELECTION;

/**
 * Created by lensenandr on 29/08/17.
 */
public class EuroGP18Analysis {
    public static final int NUM_CLUSTERS = 40;
    private static final boolean ARI = true;
    public static List<ClassifierAlg> classifierAlgs = new ArrayList<>();
    public static List<ClustererAlg> clusteringAlgs = new ArrayList<>();
    static boolean doClustering = true;

    static {
        //  classifierAlgs.add(J48::new);
        classifierAlgs.add(RandomForest::new);
        classifierAlgs.add(() -> {
            IBk ibk = new IBk();
            ibk.setKNN(3);
            return ibk;
        });
        classifierAlgs.add(NaiveBayes::new);
        classifierAlgs.add(SMO::new);
        //classifierAlgs.add(MultilayerPerceptron::new);

        clusteringAlgs.add((int seed) -> {
            SimpleKMeans kMeans = new SimpleKMeans();
            kMeans.setNumClusters(NUM_CLUSTERS);
            kMeans.setInitializationMethod(new SelectedTag(SimpleKMeans.KMEANS_PLUS_PLUS, TAGS_SELECTION));
            kMeans.setSeed(seed);
            return kMeans;
        });
        clusteringAlgs.add((int seed) -> {
            HierarchicalClusterer agglomerative = new HierarchicalClusterer();
            agglomerative.setNumClusters(NUM_CLUSTERS);
            agglomerative.setLinkType(new SelectedTag(2, HierarchicalClusterer.TAGS_LINK_TYPE));
            return agglomerative;
        });
        clusteringAlgs.add((int seed) -> {
            EM em = new EM();
            em.setNumClusters(NUM_CLUSTERS);
            em.setSeed(seed);
            return em;
        });

    }

    public static void main(String args[]) throws Exception {
        if (doClustering) {
            doClustering();
        } else {
            doClassification(null);
        }
    }

    private static void doClassification(String regexToRemove) throws Exception {
        Path datasetsDir = Paths.get("/home/lensenandr/phd/conferences/eurogp2018/originalDatasets/wine.csv.arff");
        Stream<Path> files;
        if (Files.isDirectory(datasetsDir)) {
            files = Files.list(datasetsDir);
        } else {
            files = Stream.of(datasetsDir);
        }
        Map<ClassifierAlg, List<Double>> trainRes = new LinkedHashMap<>();
        Map<ClassifierAlg, List<Double>> testRes = new LinkedHashMap<>();
        classifierAlgs.forEach(cA -> {
            trainRes.put(cA, new ArrayList<>());
            testRes.put(cA, new ArrayList<>());
        });

        files.forEach(f -> {
            try {
                Instances data = getInstances(f, regexToRemove);

                Random random = new Random(0);
                data.randomize(random);

                int trainSize = (int) Math.round(data.numInstances() * 0.7);
                int testSize = data.numInstances() - trainSize;
                Instances train = new Instances(data, 0, trainSize);
                Instances test = new Instances(data, trainSize, testSize);


                //System.out.println(data.toSummaryString());
                classifierAlgs.forEach(cA -> {
                    try {
                        classify(cA.getClassifier(), train, test, trainRes.get(cA), testRes.get(cA));
                    } catch (Exception e) {
                        throw new Error(e);
                    }
                });
            } catch (Exception e) {
                throw new Error(e);
            }
        });
        Util.LOG.println("\n============================\nFINAL RESULTS:\n");
        printResults(trainRes, testRes);


    }

    public static Instances getInstances(Path f, String regexToRemove) throws Exception {
        Instances data;
        if (f.toString().endsWith(".arff")) {
            ArffLoader arffLoader = new ArffLoader();
            arffLoader.setSource(f.toFile());
            data = arffLoader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
        } else {
            CSVLoader csvLoader = new CSVLoader();

            csvLoader.setSource(f.toFile());
            data = csvLoader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            if (data.classAttribute().isNumeric()) {
                NumericToNominal numericToNominal = new NumericToNominal();
                numericToNominal.setAttributeIndices("last-last");
                numericToNominal.setInputFormat(data);
                data = Filter.useFilter(data, numericToNominal);
            }

        }
        if (regexToRemove != null) {
            List<Integer> indiciesToRemove = new ArrayList<>();
            for (int i = 0; i < data.numAttributes(); i++) {
                if (data.attribute(i).name().matches(regexToRemove)) {
                    indiciesToRemove.add(i);
                }
            }
            Remove remove = new Remove();
            int[] toRemove = indiciesToRemove.stream().mapToInt(Integer::intValue).toArray();
            remove.setAttributeIndicesArray(toRemove);
            remove.setInputFormat(data);
            data = Filter.useFilter(data, remove);
            //   Util.LOG.println(Collections.list(data.enumerateAttributes()).stream().map(Attribute::name).collect(Collectors.joining(",")));
        }
        //Fixes silly whitespaces.
        for (int i = 0; i < data.numAttributes(); i++) {
            data.renameAttribute(i, data.attribute(i).name().trim());
        }

        return data;
    }

    public static void classify(Classifier classifier, Instances train, Instances test, List<Double> dtTrainResults, List<Double> dtTestResults) throws Exception {

        classifier.buildClassifier(train);
        Evaluation eval = new Evaluation(train);

        eval.evaluateModel(classifier, train);
        double trainAcc = eval.pctCorrect() / 100;
        eval = new Evaluation(train);
        eval.evaluateModel(classifier, test);
        double testAcc = eval.pctCorrect() / 100;
        Util.LOG.printf2("%s: Train: %.3f, Test: %.3f\n", classifier.getClass().getSimpleName(), trainAcc, testAcc);

        dtTrainResults.add(trainAcc);
        dtTestResults.add(testAcc);

    }

    static void doClustering() throws Exception {

        Path datasetsDir = Paths.get("/home/lensenandr/phd/conferences/eurogp2018/10d40cEAll");
        Stream<Path> files;
        if (Files.isDirectory(datasetsDir)) {
            files = Files.list(datasetsDir);
        } else {
            files = Stream.of(datasetsDir);
        }

        Map<ClustererAlg, List<Double>> results = new LinkedHashMap<>();
        clusteringAlgs.forEach(cA -> results.put(cA, new ArrayList<>()));


        files.forEach(f -> {
            Util.LOG.println(f);
            try {
                Instances finalData = getInstances(f, null); // "effectively final"
                clusteringAlgs.forEach(cA -> {
                    try {
                        cluster(cA, finalData, results.get(cA));
                    } catch (Exception e) {
                        throw new Error(e);
                    }
                });
            } catch (Exception e) {
                throw new Error(e);
            }
        });
        System.out.println("\n============================\nFINAL RESULTS:\n" + datasetsDir);
        for (ClustererAlg cA : clusteringAlgs) {
            List<Double> perf = results.get(cA);
            double meanPerf = Util.getMean(perf);
            double perfStdDev = Util.getStandardDeviation(perf, meanPerf);
            System.out.printf("%s: Mean ARI: %.3f+/-%.3f\n", cA.getClusterer(-1).getClass().getSimpleName(), meanPerf, perfStdDev);
        }
    }

    private static void cluster(ClustererAlg clusterer, Instances data, List<Double> results) throws Exception {
        List<Double> runs = new ArrayList<>();
        if (clusterer.getClusterer(-1) instanceof HierarchicalClusterer) {
            results.add(getSimilarity(data, clusterer, 0));
        } else {
            for (int i = 0; i < 30; i++) {
                System.out.println(i);
                runs.add(getSimilarity(data, clusterer, i));
            }
            results.add(runs.stream().mapToDouble(d -> d).average().getAsDouble());
        }
    }

    static double getSimilarity(final Instances dataSet, ClustererAlg clusterAlg, int seed) throws Exception {
        Instances noClass = removeClass(dataSet);
        // System.out.println(noClass.toSummaryString());
        AbstractClusterer clusterer = clusterAlg.getClusterer(seed);
        clusterer.buildClusterer(noClass);
        //  System.out.println(clusterer.toString());

        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(clusterer);
        eval.evaluateClusterer(noClass);
        double[] clusterAssignments = eval.getClusterAssignments();

        if (ARI) {
            ArrayModifiableDBIDs[] known = new ArrayModifiableDBIDs[NUM_CLUSTERS];
            ArrayModifiableDBIDs[] found = new ArrayModifiableDBIDs[NUM_CLUSTERS];

            for (int i = 0; i < NUM_CLUSTERS; i++) {
                known[i] = DBIDUtil.newArray();
                found[i] = DBIDUtil.newArray();
            }
            for (int i = 0; i < dataSet.size(); i++) {
                Instance instance = dataSet.get(i);
                found[(int) (clusterAssignments[i])].add(DBIDUtil.importInteger(i));
                known[(int) instance.value(dataSet.classIndex())].add(DBIDUtil.importInteger(i));
            }

            List<Cluster<Model>> knownClusters = new ArrayList<>();
            for (ArrayModifiableDBIDs arrayModifiableDBIDs : known) {
                knownClusters.add(new Cluster<>(arrayModifiableDBIDs));
            }

            List<Cluster<Model>> foundClusters = new ArrayList<>();
            for (ArrayModifiableDBIDs arrayModifiableDBIDs : found) {
                foundClusters.add(new Cluster<>(arrayModifiableDBIDs));
            }


            Clustering<Model> knownClustering = new Clustering<>("Known", "Known", knownClusters);
            Clustering<Model> foundClustering = new Clustering<>("Found", "Found", foundClusters);
            //  System.out.println(knownClustering);
            //   System.out.println(foundClustering);

            return new ClusteringFowlkesMallowsSimilarityFunction().similarity(knownClustering, foundClustering);
        } else {

            List<data.Instance> known = new ArrayList<>();
            List<List<data.Instance>> found = new ArrayList<>(NUM_CLUSTERS);

            for (int i = 0; i < NUM_CLUSTERS; i++) {
                found.add(new ArrayList<>());
            }
            for (int i = 0; i < dataSet.size(); i++) {
                Instance instance = dataSet.get(i);
                data.Instance convertedInstance = new data.Instance(instance.toDoubleArray(), "" + (int) instance.classValue(), i);

                found.get((int) clusterAssignments[i]).add(convertedInstance);
                known.add(convertedInstance);
            }

            List<clustering.Cluster> foundClusters = new ArrayList<>();
            for (List<data.Instance> instances : found) {
                CentroidCluster cluster = new CentroidCluster(clustering.Cluster.computeCentre(dataSet.numAttributes(), instances));
                cluster.addAllInstances(instances);
                foundClusters.add(cluster);
            }

//System.out.println(foundClusters);
            //      System.out.println(known);

            return PerformanceEvaluation.fMeasure(foundClusters, known);
        }
    }

    private static Instances removeClass(Instances inst) {
        Remove af = new Remove();
        Instances retI = null;

        try {
            if (inst.classIndex() < 0) {
                retI = inst;
            } else {
                af.setAttributeIndices("" + (inst.classIndex() + 1));
                af.setInvertSelection(false);
                af.setInputFormat(inst);
                retI = Filter.useFilter(inst, af);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return retI;
    }

    public static List<Future<Classifier>> tenFold(Classifier classifier, Instances instances, List<Double> trainRes, List<Double> testRes) throws Exception {
        List<Future<Classifier>> futures = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            Random random = new Random(i);
            Instances trainCV = instances.trainCV(10, i);
            trainCV.randomize(random);
            Instances testCV = instances.testCV(10, i);
            testCV.randomize(random);
            Classifier copiedClassifier = AbstractClassifier.makeCopy(classifier);

            futures.add(Util.submitJob(() -> {
                classify(copiedClassifier, trainCV, testCV, trainRes, testRes);
                return classifier;
            }));
        }
        return futures;
//        futures.forEach(f -> {
//            try {
//                f.get();
//            } catch (InterruptedException | ExecutionException e) {
//                e.printStackTrace();
//                throw new Error(e);
//            }
//        });


    }

    public static List<Map<ClassifierAlg, List<Double>>> doKFoldOnInstances(Instances wekaInstances) {

        Map<ClassifierAlg, List<Double>> trainRes = Collections.synchronizedMap(new LinkedHashMap<>());
        Map<ClassifierAlg, List<Double>> testRes = Collections.synchronizedMap(new LinkedHashMap<>());
        classifierAlgs.forEach(cA -> {
            trainRes.put(cA, Collections.synchronizedList(new ArrayList<>()));
            testRes.put(cA, Collections.synchronizedList(new ArrayList<>()));
        });

        Random random = new Random(0);
        wekaInstances.randomize(random);

        //System.out.println(data.toSummaryString());
        List<Future<Classifier>> allFutures = new ArrayList<>();
        classifierAlgs.forEach(cA -> {
            try {
                List<Future<Classifier>> futures = tenFold(cA.getClassifier(), wekaInstances, trainRes.get(cA), testRes.get(cA));
                allFutures.addAll(futures);
            } catch (Exception e) {
                throw new Error(e);
            }
        });

        allFutures.forEach(f -> {
            try {
                f.get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
                throw new Error(e);
            }
        });


        Util.LOG.println("============================\nFINAL RESULTS:\n");
        printResults(trainRes, testRes);
        List<Map<ClassifierAlg, List<Double>>> bothResults = new ArrayList<>();
        bothResults.add(trainRes);
        bothResults.add(testRes);
        return bothResults;
    }

    public static void printResults(Map<ClassifierAlg, List<Double>> trainRes, Map<ClassifierAlg, List<Double>> testRes) {
        for (ClassifierAlg cA : classifierAlgs) {
            List<Double> trainPerf = trainRes.get(cA);
            List<Double> testPerf = testRes.get(cA);
            printResults(cA, trainPerf, testPerf);
        }
    }

    public static void printResults(ClassifierAlg cA, List<Double> trainPerf, List<Double> testPerf) {
        double trainMean = Util.getMean(trainPerf);
        double trainStdDev = Util.getStandardDeviation(trainPerf, trainMean);
        double testMean = Util.getMean(testPerf);
        double testStdDev = Util.getStandardDeviation(testPerf, testMean);
        Util.LOG.printf("%s: Train: %.3f+/-%.3f, Test: %.3f+/-%.3f\n", cA.getClassifier().getClass().getSimpleName(), trainMean, trainStdDev, testMean, testStdDev);
    }

    @FunctionalInterface
    public interface ClassifierAlg {
        Classifier getClassifier();
    }

    @FunctionalInterface
    public interface ClustererAlg {
        AbstractClusterer getClusterer(int seed) throws Exception;
    }
}
