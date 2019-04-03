package featureGrouping;

import infodynamics.measures.continuous.kraskov.MultiInfoCalculatorKraskov1;
import infodynamics.measures.continuous.kraskov.MutualInfoCalculatorMultiVariateKraskov;
import infodynamics.measures.continuous.kraskov.MutualInfoCalculatorMultiVariateKraskov1;
import infodynamics.measures.continuous.kraskov.MutualInfoCalculatorMultiVariateKraskov2;
import other.Util;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

/**
 * Created by lensenandr on 10/08/17.
 */
public class MutualInformationMap implements VariableDependencyMap, Serializable {
    private static final boolean addNoise = true;
    private static final boolean addNoiseMultiInfo = true;
    private static final boolean addNoiseMultivariate = true;
    private final double[][] mu;

    public MutualInformationMap(List<ValuedFeature> valuedFeatures) {
        int n = valuedFeatures.size();
        mu = new double[n][n];
        List<Future<?>> futures = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int finalI = i;

            Future<Void> future = Util.submitJob(() -> {
                ValuedFeature vf = valuedFeatures.get(finalI);
                //By definition.
                mu[vf.featureID][vf.featureID] = getMutualInformation(vf, vf);
                System.out.printf("M%.2f%%\n", vf.featureID * 100 / (double) valuedFeatures.size());

                for (int j = finalI + 1; j < n; j++) {
                    ValuedFeature otherFeature = valuedFeatures.get(j);
                    double mic = getMutualInformation(vf, otherFeature);
                    //  System.out.println(mic);
                    mu[vf.featureID][otherFeature.featureID] = mic;
                    mu[otherFeature.featureID][vf.featureID] = mic;
                }
                return null;
            });

            futures.add(future);
        }

        for (Future<?> future : futures) {
            try {
                future.get();
            } catch (InterruptedException | ExecutionException e) {
                throw new Error(e);
            }
        }

    }

    public static double getMutualInformation(ValuedFeature x, ValuedFeature y) {
        return getMutualInformation(x.values, y.values);
    }

    public static double getMutualInformation(final double[] source, final double[] target) {
        return getMutualInformation(source, target, addNoise);
    }

    public static double getMutualInformation(final double[] source, final double[] target, boolean noise) {
        MutualInfoCalculatorMultiVariateKraskov1 miCalculator = new MutualInfoCalculatorMultiVariateKraskov1();
        try {
            miCalculator.setProperty(MutualInfoCalculatorMultiVariateKraskov.PROP_ADD_NOISE, "false");
            miCalculator.setProperty(MutualInfoCalculatorMultiVariateKraskov.PROP_NUM_THREADS, "1");

            //  miCalculator.setProperty(MutualInfoCalculatorMultiVariateKraskov.PROP_NORMALISE,"false");
            double[] thisSource = new double[source.length];
            System.arraycopy(source, 0, thisSource, 0, source.length);
            double[] thisTarget = new double[target.length];
            System.arraycopy(target, 0, thisTarget, 0, target.length);

            if (noise) {
                Random random = new Random( /*TODO*/);
                // Add Gaussian noise of std dev noiseLevel to the data
                for (int r = 0; r < thisSource.length; r++) {
                    thisSource[r] += random.nextGaussian() * 1e-8;
                    thisTarget[r] += random.nextGaussian() * 1e-8;
                }
            }

            miCalculator.setObservations(thisSource, thisTarget);
            return miCalculator.computeAverageLocalOfObservations();
        } catch (Exception e) {
            e.printStackTrace();
            throw new Error(e);
        }


    }

    public static double getMultiVarMutualInformation(final double[][] source, final double[][] target) {
        return getMultiVarMutualInformation(source, target, addNoiseMultivariate);
    }

    public static double getMultiVarMutualInformation(final double[][] source, final double[][] target, boolean noise) {
//        if (source.length != target.length) {
//            throw new IllegalArgumentException("Diff num of source and targets");
//        }
        int sourceDim = source[0].length;
        int targetDim = target[0].length;
        if (sourceDim != targetDim) {
            throw new IllegalArgumentException("Source and target dimensionalities differ");
        }

        MutualInfoCalculatorMultiVariateKraskov1 miCalculator = new MutualInfoCalculatorMultiVariateKraskov1();
        try {
            miCalculator.setProperty(MutualInfoCalculatorMultiVariateKraskov.PROP_ADD_NOISE, "false");
            miCalculator.setProperty(MutualInfoCalculatorMultiVariateKraskov.PROP_NUM_THREADS, "1");
            //TODO
            //  miCalculator.setProperty(MutualInfoCalculatorMultiVariateKraskov.PROP_K,"1");

            //  miCalculator.setProperty(MutualInfoCalculatorMultiVariateKraskov.PROP_NORMALISE,"false");

            //Transpose it all, since they have rows as instances, not features
            double[][] thisSource = transposeMatrix(source, noise);
            Random random;
            double[][] thisTarget = transposeMatrix(target, noise);

            miCalculator.initialise(thisSource[0].length, thisTarget[0].length);
            miCalculator.setObservations(thisSource, thisTarget);
            double v = miCalculator.computeAverageLocalOfObservations();
            // System.err.println(v);
            return v;
        } catch (Exception e) {
            e.printStackTrace();
            throw new Error(e);
        }


    }


    public static double getMultiVarMutualInformationVers2(final double[][] source, final double[][] target) {
//        if (source.length != target.length) {
//            throw new IllegalArgumentException("Diff num of source and targets");
//        }
        int sourceDim = source[0].length;
        int targetDim = target[0].length;
        if (sourceDim != targetDim) {
            throw new IllegalArgumentException("Source and target dimensionalities differ");
        }

        MutualInfoCalculatorMultiVariateKraskov miCalculator = new MutualInfoCalculatorMultiVariateKraskov2();
        try {
            miCalculator.setProperty(MutualInfoCalculatorMultiVariateKraskov.PROP_ADD_NOISE, "false");
            miCalculator.setProperty(MutualInfoCalculatorMultiVariateKraskov.PROP_NUM_THREADS, "1");

            double[][] thisSource = transposeMatrix(source, false);
            Random random;
            double[][] thisTarget = transposeMatrix(target, false);

            miCalculator.initialise(thisSource[0].length, thisTarget[0].length);
            miCalculator.setObservations(thisSource, thisTarget);
            double v = miCalculator.computeAverageLocalOfObservations();
            // System.err.println(v);
            return v;
        } catch (Exception e) {
            e.printStackTrace();
            throw new Error(e);
        }


    }

    public static double[][] transposeMatrix(double[][] matrix, boolean noise) {
        double[][] thisSource = new double[matrix[0].length][matrix.length];
        Random random = new Random( /*TODO*/);

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                thisSource[j][i] = matrix[i][j] + (noise ? random.nextGaussian() * 1e-8 : 0);
            }
        }
        return thisSource;
    }


    public static double getMultiInformation(final double[][] set) {
        return getMultiInformation(set, addNoiseMultiInfo);
    }

    public static double getMultiInformation(final double[][] set, boolean noise) {
        //TODO: fix this --- swap rows and columns!!!
        //ProcessBuilder processBuilder = new ProcessBuilder("python","/home/lensenandr/IdeaProjects/phd/src/doMI.py",)
        MultiInfoCalculatorKraskov1 miCalculator = new MultiInfoCalculatorKraskov1();
        try {
            miCalculator.setProperty(MultiInfoCalculatorKraskov1.PROP_ADD_NOISE, "false");
            miCalculator.setProperty(MultiInfoCalculatorKraskov1.PROP_NUM_THREADS, "1");

            double[][] thisSet = transposeMatrix(set, noise);

            miCalculator.initialise(thisSet[0].length);
            miCalculator.setObservations(thisSet);
            return miCalculator.computeAverageLocalOfObservations();

        } catch (Exception e) {
            e.printStackTrace();
            throw new Error(e);
        }


    }


    @Override
    public double getDependency(ValuedFeature vf1, ValuedFeature vf2) {
        return mu[vf1.featureID][vf2.featureID];
    }
}
