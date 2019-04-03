package other;

import data.Instance;
import data.UnlabelledInstance;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import static java.util.Arrays.stream;
import static java.util.stream.Collectors.joining;
import static java.util.stream.Collectors.toList;
import static other.Util.*;

/**
 * Created by lensenandr on 22/03/16.
 */
public class DatasetUtils {

    public static boolean[] ALL_FEATURES;
    public static int FEATURE_MIN;
    public static int FEATURE_MAX;
    public static double FEATURE_RANGE;
    public static double FEATURE_RANGE2;


    public static List<Instance> getInstances(List<String> lines, String classLabelFirst, int numFeatures, String splitChar) {
        int numInstances = lines.size() - 1;
        boolean hasClassLabel = !classLabelFirst.equalsIgnoreCase("noClass");
        boolean oneOffset = classLabelFirst.equalsIgnoreCase("classFirst");
        List<Instance> instances = new ArrayList<>(numInstances);
        double[][] input = new double[numInstances][numFeatures];
        String[] labels = new String[numInstances];
        for (int lineNum = 1; lineNum < lines.size(); lineNum++) {
            String line = lines.get(lineNum).trim();
            String[] split = line.split(splitChar);


            if (split.length == (hasClassLabel ? numFeatures + 1 : numFeatures)) {
                for (int i = 0; i < numFeatures; i++) {
                    double featureValue = Util.toDouble(split[oneOffset ? i + 1 : i]);
                    input[lineNum - 1][i] = featureValue;
                }
                if (hasClassLabel)
                    labels[lineNum - 1] = split[oneOffset ? 0 : numFeatures];
            }
        }
        boolean[] validFeatures = new boolean[numFeatures];
        int numValidFeatures = 0;
        for (int i = 0; i < numFeatures; i++) {
            double min = Double.MAX_VALUE;
            double max = MOST_NEGATIVE_VAL;
            for (int j = 0; j < numInstances; j++) {
                double val = input[j][i];
                if (val < min) min = val;
                if (val > max) max = val;
            }
            if (min != max) {
                validFeatures[i] = true;
                numValidFeatures++;
            } else {
                LOG.printf("Ignoring feature %d, has 0 variance.\n", i);
            }
        }
        for (int i = 0; i < numInstances; i++) {
            double[] featureVals = new double[numValidFeatures];
            int index = 0;
            for (int j = 0; j < validFeatures.length; j++) {
                if (validFeatures[j]) {
                    featureVals[index] = input[i][j];
                    index++;
                }

            }
            if (index != numValidFeatures) {
                throw new IllegalStateException();
            }
            if (labels[i] != null) {
                instances.add(new Instance(featureVals, labels[i], i));
            } else {
                throw new IllegalArgumentException();
                // instances.add(new UnlabelledInstance(featureVals));
            }
        }
        return instances;

    }

    public static List<Instance> scaleInstances(List<Instance> instances) {
        int numFeatures = instances.get(0).numFeatures();
        double[] minFeatureVals = new double[numFeatures];
        Arrays.fill(minFeatureVals, Double.MAX_VALUE);
        double[] maxFeatureVals = new double[numFeatures];
        Arrays.fill(maxFeatureVals, MOST_NEGATIVE_VAL);

        for (Instance instance : instances) {
            for (int i = 0; i < numFeatures; i++) {
                double featureValue = instance.getFeatureValue(i);
                if (featureValue < minFeatureVals[i]) {
                    minFeatureVals[i] = featureValue;
                }
                if (featureValue > maxFeatureVals[i]) {
                    maxFeatureVals[i] = featureValue;
                }
            }
        }
        //    Util.LOG.println2("Min vals: " + Arrays.toString(minFeatureVals));
        //  Util.LOG.println2("Max vals: " + Arrays.toString(maxFeatureVals));

        return instances.stream().map(instance -> instance.scaledCopy(minFeatureVals, maxFeatureVals)).collect(toList());
    }

    public static void scaleArray(double[] vals) {
        int numVals = vals.length;
        double minFeatureVal = Double.MAX_VALUE;
        double maxFeatureVal = MOST_NEGATIVE_VAL;


        for (int i = 0; i < numVals; i++) {
            double val = vals[i];
            if (val < minFeatureVal) {
                minFeatureVal = val;
            }
            if (val > maxFeatureVal) {
                maxFeatureVal = val;
            }
        }
        for (int i = 0; i < numVals; i++) {
            vals[i] = scale(vals[i], minFeatureVal, maxFeatureVal);

        }
    }

    public static double[] normaliseFeature(double[] featureValues) {
        //Standard score. (value - mean)/stdDev
        double mean = stream(featureValues).average().getAsDouble();

        double sumVariance = stream(featureValues).map(val -> val - mean).map(diff -> (diff * diff)).sum();
        double stdDev = Math.sqrt(sumVariance / (featureValues.length - 1));

        return stream(featureValues).map(val -> (val - mean) / stdDev).toArray();

    }


    public static List<Instance> normaliseInstances(List<Instance> instances) {
        //Standard score. (value - mean)/stdDev
        int numInstances = instances.size();
        int numFeatures = instances.get(0).numFeatures();
        double[] featureMeans = getFeatureMeans(instances, numInstances, numFeatures);

        double[] featureStdDevs = getFeatureStandardDeviations(instances, numInstances, numFeatures, featureMeans);

        return instances.stream().map(instance -> instance.normalisedCopy(featureMeans, featureStdDevs)).collect(toList());

    }


    private static double[] getFeatureStandardDeviations(List<Instance> instances, int numInstances, int numFeatures, double[] featureMeans) {
        double[] featureStdDevs = new double[numFeatures];

        for (Instance instance : instances) {
            for (int i = 0; i < numFeatures; i++) {
                double diff = instance.getFeatureValue(i) - featureMeans[i];
                featureStdDevs[i] += (diff * diff);
            }

        }
        for (int i = 0; i < featureStdDevs.length; i++) {
            featureStdDevs[i] = Math.sqrt(featureStdDevs[i] / numInstances);

        }
        return featureStdDevs;
    }

    private static double[] getFeatureMeans(List<Instance> instances, int numInstances, int numFeatures) {
        double[] featureMeans = new double[numFeatures];

        for (Instance instance : instances) {
            for (int i = 0; i < numFeatures; i++) {
                double featureValue = instance.getFeatureValue(i);
                featureMeans[i] += featureValue;
            }

        }
        for (int i = 0; i < featureMeans.length; i++) {
            featureMeans[i] /= numInstances;

        }
        return featureMeans;
    }



    public static int numFeaturesUsed(boolean[] featureSubset) {
        int numFeaturesUsed = 0;
        for (boolean b : featureSubset) {
            if (b) numFeaturesUsed++;

        }
        return numFeaturesUsed;
    }


    /**
     * Bad design. Pls fix.
     *
     * @param instances
     */
    public static void initialise(List<Instance> instances, boolean doNNs) {
        int numFeatures = instances.get(0).numFeatures();
        ALL_FEATURES = new boolean[numFeatures];
        Arrays.fill(ALL_FEATURES, true);


    }


}
