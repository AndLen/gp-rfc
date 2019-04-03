package featureCreate.gp;

import ec.EvolutionState;
import ec.Individual;
import ec.Statistics;
import ec.gp.GPIndividual;
import ec.gp.GPTree;
import ec.simple.SimpleFitness;
import featureCreate.FeatureCreator;
import gp.DoubleData;
import gp.MIFitness;

import java.util.Arrays;

import static other.Util.LOG;

/**
 * Created by lensenandr on 8/07/16.
 */
public class FeatureCreatorStatistics extends Statistics {


    public static void printStatistics(double[][] outputs, double fitness) {
        LOG.printf("Best ind: fitness %.3f\n", fitness);
        // LOG.printf("Inputs: %s", format(FeatureCreator.xVals));

        if (FeatureCreator.xVals != null) {
            OutputEvaluator oE = new OutputEvaluator(outputs).invoke();
            double worstSourceMI = oE.getWorstSourceMI();
            double worstSharedMI = oE.getWorstSharedMI();

            LOG.printf("MAX Source: %.2f, Mean Source: %.2f, min Source: %.2f; MAX Shared: %.2f, Mean Shared: %.2f, min SHared: %.2f\n",
                    oE.maxSourceMI, oE.meanSourceMI, oE.minSourceMI, oE.maxSharedMI, oE.meanSharedMI, oE.minSharedMI);
            //LOG.printf("Worst Source/Worst Shared: %.3f\n", minSourceMI / maxSharedMI);
            LOG.printf("Worst Source-Worst Shared: %.3f\n", worstSourceMI - worstSharedMI);

            double multiInfo = MTFCTreeSimilarityProblem.getMultiInfo(outputs);
            LOG.printf("Multi-Info: %.3f\n", multiInfo);
        }
        //   if (worstSourceMI >= 0) {
        if (FeatureCreator.multipleXVals != null) {
            LOG.printf("Multivariate MI: %.3f\n", MTFCMultivariateProblem.getMutualInfo(outputs));

        }

        double[] minDifferences = new double[outputs.length];
        for (int i = 0; i < outputs.length; i++) {
            minDifferences[i] = MTFCTreeSimilarityProblem.getMinDifference(i, outputs);
        }

        StringBuilder sbDist = new StringBuilder("[");
        for (double minDifference : minDifferences) {
            sbDist.append(String.format("%.4f,", minDifference));
        }
        sbDist.deleteCharAt(sbDist.length() - 1);
        sbDist.append("]");

        LOG.printf("Min distances between trees: %.4f\n Per tree: %s\n", Arrays.stream(minDifferences).min().getAsDouble(), sbDist.toString());
        // }}
    }

    static double printDetails(int i, double[] output, double[] sharedMI, double sourceMI) {
        double sum = 0;
        for (int j = 0; j < sharedMI.length; j++) {
            double v = sharedMI[j];
            if (!Double.isNaN(v)) {
                sum += v;
            }
        }
        sum /= sharedMI.length - 1;
        LOG.printf("F%s%c Source MI:%.2f Shared MI:%.2f\n", FeatureCreator.SOURCE_PREFIX, getCharToUse(i), sourceMI, sum);
        // LOG.printf("Output: %s", format(output));
        LOG.printf("Shared MIs: %s", format(sharedMI));
        return sum;
    }

    public static char getCharToUse(int i) {
        return i < 26 ? (char) ('a' + i) : (char) ('α' + (i - 26));
    }

    static String format(double[] outputs) {
        if (outputs == null) {
            return "[NULL]\n";
        } else {
            StringBuilder sb = new StringBuilder("[");
            sb.append(toCSV(outputs));
            sb.append("]\n");

            return sb.toString();
        }
    }

    public static String toCSV(double[] outputs) {
        StringBuilder sb = new StringBuilder();
        if (outputs == null) {
            return "NULL, ";
        } else {
            for (int i = 0; i < outputs.length - 1; i++) {
                double output = outputs[i];
                sb.append(String.format("%f, ", output));
            }
            sb.append(String.format("%f", outputs[outputs.length - 1]));
            return sb.toString();
        }
    }

    public void postEvaluationStatistics(final EvolutionState state) {
        super.postEvaluationStatistics(state);

        // for now we just print the best fitness per subpopulation.
        Individual individual = null;  // quiets compiler complaints
        int subpopIndex = -1;
        for (int x = 0; x < state.population.subpops.size(); x++) {
            for (int y = 0; y < state.population.subpops.get(x).individuals.size(); y++) {
                if (state.population.subpops.get(x).individuals.get(y) != null) {
                    if (individual == null || state.population.subpops.get(x).individuals.get(y).fitness.betterThan(individual.fitness)) {
                        individual = state.population.subpops.get(x).individuals.get(y);
                        subpopIndex = x;
                    }
                }

            }
        }

        if (individual.fitness instanceof MIFitness) {
            MIFitness miFitness = (MIFitness) individual.fitness;
            Individual[] individuals = Arrays.copyOf(miFitness.context, miFitness.context.length);
            individuals[subpopIndex] = individual;

            double[][] outputs = new double[individuals.length][FeatureCreator.xVals.length];

            for (int i = 0; i < individuals.length; i++) {
                outputs[i] = ((FeatureCreatorProblem) state.evaluator.p_problem).getOutputs(state, 0, new DoubleData(), (GPIndividual) individuals[i], FeatureCreator.noisyXVals);
            }

            double fitness = miFitness.fitness();
            printStatistics(outputs, fitness);


        } else {// if (((GPIndividual) individual).trees.length > 1) {

            GPTree[] trees = ((GPIndividual) individual).trees;
            double[][] outputs = ((FeatureCreatorProblem) state.evaluator.p_problem).getAllOutputs(state, 0, (GPIndividual) individual);

            SimpleFitness simpleFitness = (SimpleFitness) individual.fitness;
            double fitness = simpleFitness.fitness();
            printStatistics(outputs, fitness);

        }
//        else {
//            double[] outputs = ((FeatureCreatorProblem) state.evaluator.p_problem).getOutputs(state, 0, new DoubleData(), (GPIndividual) individual, FeatureCreator.noisyXVals);
//            LOG.printf("Inputs: %s", format(FeatureCreator.xVals));
//            LOG.printf("Outputs: %s", format(outputs));
//            LOG.printf("Multiples: %s", formatMultiples(FeatureCreator.xVals, outputs));
//            LOG.printf("Absolute pearsons: %.2f\n", PearsonCorrelationMap.getAbsolutePearsonCorrelation(FeatureCreator.xVals, outputs));
//        }
    }

    private String formatMultiples(double[] inputs, double[] outputs) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < outputs.length - 1; i++) {
            double output = outputs[i] / inputs[i];
            sb.append(String.format("%.2f, ", output));
        }
        sb.append(String.format("%.2f]\n", outputs[outputs.length - 1] / inputs[outputs.length - 1]));

        return sb.toString();
    }

    public static class OutputEvaluator {
        double minSourceMI;
        double maxSharedMI;
        double maxSourceMI;
        double meanSourceMI;
        double minSharedMI;
        double meanSharedMI;
        private double[][] outputs;
        private int length;

        public OutputEvaluator(double[][] outputs) {
            this.outputs = outputs;
            this.length = outputs.length;

        }

        public double getWorstSourceMI() {
            return minSourceMI;
        }

        public double getWorstSharedMI() {
            return maxSharedMI;
        }

        public OutputEvaluator invoke() {
            double[] sourceMIs = new double[length];

            for (int i = 0; i < length; i++) {
                if (outputs[i] == null) {
                    sourceMIs[i] = -Double.MAX_VALUE;
                } else {
                    sourceMIs[i] = FeatureCreatorProblem.getMI(FeatureCreator.xVals, outputs[i]) / FeatureCreator.baseMI;
                }
            }

            double[][] pairwiseMIs = new double[length][length];
            for (int i = 0; i < length; i++) {
                pairwiseMIs[i][i] = Double.NaN;
                for (int j = 0; j < length; j++) {
                    if (i != j) {
                        if (outputs[i] != null && outputs[j] != null) {
                            pairwiseMIs[i][j] = MultitreeFeatureCreatorProblem.getMI(outputs[i], outputs[j]);
                            pairwiseMIs[i][j] /= FeatureCreator.baseMI;
                        } else {
                            pairwiseMIs[i][j] = Double.NaN;
                        }
                    }
                }
            }
            double[] meanMIs = new double[length];
            for (int i = 0; i < length; i++) {
                meanMIs[i] = printDetails(i, outputs[i], pairwiseMIs[i], sourceMIs[i]);
            }
            minSourceMI = Arrays.stream(sourceMIs).min().getAsDouble();
            maxSourceMI = Arrays.stream(sourceMIs).max().getAsDouble();
            meanSourceMI = Arrays.stream(sourceMIs).average().getAsDouble();
            maxSharedMI = Arrays.stream(meanMIs).max().getAsDouble();
            minSharedMI = Arrays.stream(meanMIs).min().getAsDouble();
            meanSharedMI = Arrays.stream(meanMIs).average().getAsDouble();
            return this;
        }
    }
}
