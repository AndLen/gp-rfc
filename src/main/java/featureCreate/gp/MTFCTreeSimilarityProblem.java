package featureCreate.gp;

import ec.gp.GPTree;
import featureCreate.FeatureCreator;
import featureGrouping.MutualInformationMap;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalDouble;


/**
 * Created by lensenandr on 8/08/16.
 */
public class MTFCTreeSimilarityProblem extends MultitreeFeatureCreatorProblem {

    private static final boolean DIST_TO_SOURCE = true;


    private static double getKS(double[] subject, double[] target) {
        return 1 - new KolmogorovSmirnovTest().kolmogorovSmirnovStatistic(subject, target);
    }

    private static double minDistanceOfDifferences(double[] subject, double[][] targets) {
        double min = Double.MAX_VALUE;
        for (double[] target : targets) {
            min = Double.min(min, distanceOfDifferences(subject, target));
        }
        return min;
    }

    private static double distanceOfDifferences(double[] subject, double[] target) {
        Integer[][] sortedSourceIndicies = FeatureCreator.getSortedSourceIndicies();
        double[] meanDiffs = new double[sortedSourceIndicies.length];
        for (int i = 0; i < sortedSourceIndicies.length; i++) {
            Integer[] sortedOriginal = sortedSourceIndicies[i];
            meanDiffs[i] = distancesOfDifferencesSorted(subject, target, sortedOriginal);
        }
        //The closest according to any given source feature
        return Arrays.stream(meanDiffs).min().orElse(0d);


    }

    private static double distancesOfDifferencesSorted(double[] subject, double[] target, Integer[] sortedSourceIndicies) {
        double[] subject2 = new double[subject.length];
        // Arrays.copyOf(subject, subject.length);
        double[] target2 = new double[target.length];
        // Arrays.copyOf(target, target.length);
        for (int i = 0; i < subject.length; i++) {
            subject2[i] = subject[sortedSourceIndicies[i]];
            target2[i] = target[sortedSourceIndicies[i]];
        }

        //      System.out.println(Arrays.toString(indicies));
        //    System.out.println(Arrays.toString(subject2));
        //    List<Double> subjectGradients = new ArrayList<>(subject2.length);
        //  List<Double> targetGradients = new ArrayList<>(subject2.length);
        List<Double> localGradients = new ArrayList<>(subject2.length);
        // List<Double> localAcceleration = new ArrayList<>(subject2.length);
        // List<Double> absLocalGradients = new ArrayList<>(subject2.length);
        // List<Double> neighbourGradients = new ArrayList<>(subject2.length);

        // List<Double> absNeighbourGradients = new ArrayList<>(subject2.length);

        for (int i = 1; i < subject2.length; i++) {
            double subGradient = subject2[i] - subject2[i - 1];
            double tarGradient = target2[i] - target2[i - 1];
            localGradients.add(Math.abs(subGradient - tarGradient));
            //    double absSubGradient = Math.abs(subGradient);
            //  double absTarGradient = Math.abs(tarGradient);
            //     subjectGradients.add(absSubGradient);
            //   targetGradients.add(absTarGradient);
            // absLocalGradients.add(Math.abs(absSubGradient - absTarGradient));
//            if (i >= 2) {
//                localAcceleration.add(Math.abs(localGradients.get(localGradients.size() - 1) - localGradients.get(localGradients.size() - 2)));
//                if (i < subject2.length - 2) {
//                    neighbourGradients.add(Math.abs((subject2[i + 2] - subject2[i - 2]) - (target2[i + 2] - subject2[i - 2])));
//                    absNeighbourGradients.add(Math.abs(Math.abs(subject2[i + 2] - subject2[i - 2]) - Math.abs(target2[i + 2] - subject2[i - 2])));
//                }
//            }
        }
        //     double subGradientStdDev = Util.getStandardDeviation(subjectGradients, subjectGradients.stream().mapToDouble(d -> d).average().getAsDouble());
        //   double tarGradientStdDev = Util.getStandardDeviation(targetGradients, targetGradients.stream().mapToDouble(d -> d).average().getAsDouble());
        double result = localGradients.stream().mapToDouble(d -> d).average().getAsDouble();
        // System.out.println(result);
        return //Math.abs(subGradientStdDev - tarGradientStdDev);
                result;// / (subject2.length - 1);
    }

    static double getMultiInfo(double[][] outputs) {
        double[][] sourceAndTargets = new double[outputs.length + 1][];
        sourceAndTargets[0] = FeatureCreator.xVals;
        for (int i = 0; i < outputs.length; i++) {
            sourceAndTargets[i + 1] = outputs[i];
        }


        return MutualInformationMap.getMultiInformation(sourceAndTargets) / FeatureCreator.baseMultiInfo;
    }

    public static double getMinDifference(int i, double[][] outputs) {
        double worst = Double.MAX_VALUE;
        if (outputs[i] != null) {
            double[] subject = outputs[i];
            for (int j = 0; j < outputs.length; j++) {
                if (i != j) {
                    if (outputs[i] != null) {
                        double distance = distanceOfDifferences(subject, outputs[j]);
                        if (distance < worst) {
                            worst = distance;
                        }
                    }
                }

            }
            if (DIST_TO_SOURCE) {
                double distToSource;
                if (FeatureCreator.MULTIVARIATE) {
                    double minDist = Double.MAX_VALUE;
                    double[][] multipleXVals = FeatureCreator.multipleXVals;
                    for (int i1 = 0; i1 < multipleXVals.length; i1++) {
                        double[] thisSource = multipleXVals[i1];
                        minDist = Double.min(minDist, distancesOfDifferencesSorted(subject, thisSource, FeatureCreator.getSortedSourceIndicies()[i1]));

                    }
                    distToSource = minDist;
                } else {
                    distToSource = distanceOfDifferences(subject, FeatureCreator.xVals);
                }
                if (distToSource < worst) {
                    worst = distToSource;
                }
            }
        }
        return worst;
    }

    static double[] getMinDistBetweenOutputs(double[][] outputs) {
        int numFeatures;
        numFeatures = outputs.length;
        double minDifferences[] = new double[numFeatures];


        for (int i = 0; i < numFeatures; i++) {
            //       meanMIs[i] = getMeanKS(i, outputs) / FeatureCreator.baseMI;
            minDifferences[i] = getMinDifference(i, outputs);
        }
        return minDifferences;
    }

    @Override
    public double internalMeasureFitness(double[][] outputs, GPTree[] trees) {
        double fitness;
        //    try {
        int numFeatures = outputs.length;

        double sourceMIs[] = new double[numFeatures];

        for (int i = 0; i < numFeatures; i++) {
            if (outputs[i] == null) {
                //Okay, so not a valid solution. fitness will be -ve numer of invalid trees.
                int count = 0;
                for (int j = i; j < numFeatures; j++) {
                    if (outputs[j] == null) {
                        count++;
                    }
                }
                return -count;

                //throw new CantComputeMIException();
            }
            sourceMIs[i] = getMI(FeatureCreator.xVals, outputs[i]) / FeatureCreator.baseMI;
        }

        double[] minDifferences = getMinDistBetweenOutputs(outputs);

        double multiInfo = getMultiInfo(outputs);
        //Arrays.stream(sourceMIs).min().getAsDouble();
        double worstSourceMI = Arrays.stream(sourceMIs).min().getAsDouble();
        if (worstSourceMI < MIN_SOURCE_MI) {
            //Encourage getting sourceMIs up as a priority
            return -worstSourceMI;
            //-(1 / Arrays.stream(sourceMIs).average().getAsDouble());
        }

        if (CAP_SOURCE_MI) {
            worstSourceMI = Math.min(worstSourceMI, 0.9);
        }
        OptionalDouble worst = Arrays.stream(minDifferences).min();//.getAsDouble();
        double worstDifference = worst.isPresent() && !Double.isNaN(worst.getAsDouble()) ? worst.getAsDouble() : 0;
        fitness = ((multiInfo) * worstDifference);//+0.1*worstSourceMI;

        //if (Double.isNaN(fitness)) {
        //System.err.println();
        //}
        // } catch (CantComputeMIException e) {
        //     fitness = -Double.MAX_VALUE;
        //}
        return fitness;
    }

    private double getWorstKS(int i, double[][] outputs) {
        double worst = -Double.MAX_VALUE;
        double[] subject = outputs[i];
        for (int j = 0; j < outputs.length; j++) {
            if (i != j) {

                double ks = getKS(subject, outputs[j]);
                if (ks > worst) {
                    worst = ks;
                }
            }

        }
        return worst;
    }

    private double getMeanKS(int i, double[][] outputs) {
        double sum = 0;
        double[] subject = outputs[i];
        for (int j = 0; j < outputs.length; j++) {
            if (i != j) {
                sum += getKS(subject, outputs[j]);
            }

        }
        return sum / (outputs.length - 1);
    }

}


