package featureCreate.gp;

import ec.EvolutionState;
import ec.Individual;
import ec.gp.GPIndividual;
import ec.gp.GPProblem;
import ec.gp.GPTree;
import ec.simple.SimpleFitness;
import featureCreate.FeatureCreator;
import featureGrouping.MutualInformationMap;
import gp.DoubleData;
import other.DatasetUtils;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.OptionalDouble;

/**
 * Created by lensenandr on 29/08/17.
 */
public class FeatureCreatorProblem extends GPProblem implements FCProblemInterface {

    public double x;

    public static double getMI(double[] source, double[] results) {
        double fitness = MutualInformationMap.getMutualInformation(source, results);
        //TODO: Prevent 0s...
        return Math.max(fitness, 0);
    }

    public static double getMVMI(double[][] source, double[][] results) {
        double fitness = MutualInformationMap.getMultiVarMutualInformation(source, results);
        //TODO: Prevent 0s...
        return Math.max(fitness, 0);
    }

    public static double measureFitness(double[][] outputs) {
        double fitness;
        try {
            int numFeatures = outputs.length;

            double sourceMIs[] = new double[numFeatures];

            for (int i = 0; i < numFeatures; i++) {
                if (outputs[i] == null) {
                    throw new CantComputeMIException();
                }
                sourceMIs[i] = getMI(outputs[i]) / FeatureCreator.baseMI;
            }
            double meanMIs[] = new double[numFeatures];

            for (int i = 0; i < numFeatures; i++) {
                meanMIs[i] = getMeanSharedMI(i, outputs) / FeatureCreator.baseMI;
            }

            double worstSourceMI = Arrays.stream(sourceMIs).min().getAsDouble();
            if (worstSourceMI < MIN_SOURCE_MI) {
                //Encourage getting sourceMIs up as a priority
                return -(1 / Arrays.stream(sourceMIs).average().getAsDouble());
            }

            if (CAP_SOURCE_MI) {
                worstSourceMI = Math.min(worstSourceMI, 0.9);
            }
            OptionalDouble max = Arrays.stream(meanMIs).max();//.getAsDouble();
            double worstSharedMI = max.isPresent() && !Double.isNaN(max.getAsDouble()) ? max.getAsDouble() : 1;
            fitness = ((worstSourceMI) - worstSharedMI);//+0.1*worstSourceMI;

            if (Double.isNaN(fitness)) {
                //System.err.println();
            }
        } catch (CantComputeMIException e) {
            fitness = -Double.MAX_VALUE;
        }
        return fitness;
    }

    public static double getMeanSharedMI(int i, double[][] outputs) {
        double sum = 0;
        double[] subject = outputs[i];
        for (int j = 0; j < outputs.length; j++) {
            if (i != j) {
                sum += getMI(subject, outputs[j]);
            }

        }
        return sum / (outputs.length - 1);
    }

    static double getMI(double[] results) throws CantComputeMIException {
        double[] xVals = FeatureCreator.xVals;
        if (results == null) {
            throw new CantComputeMIException();
        }
        return getMI(xVals, results);
    }

    public double internalMeasureFitness(double[][] outputs, GPTree[] trees) {
        return measureFitness(outputs);
    }

    @Override
    public void evaluate(EvolutionState state, Individual ind, int subpopulation, int threadnum) {
        if (!ind.evaluated)  // don't bother reevaluating
        {
            DoubleData input = (DoubleData) (this.input);
            GPIndividual gpInd = (GPIndividual) ind;
            double fitness;
            try {
                fitness = getMI(getOutputs(state, threadnum, input, gpInd));
            } catch (CantComputeMIException e) {
                fitness = -Double.MAX_VALUE;
            }
            SimpleFitness f = ((SimpleFitness) ind.fitness);

            if (Math.abs(fitness - FeatureCreator.baseMI) < 0.0001) {
                f.setFitness(state, 0, false);
            } else {
                f.setFitness(state, fitness / FeatureCreator.baseMI, false);
            }
            ind.evaluated = true;

        }

    }

    public double[] getOutputs(EvolutionState state, int threadnum, DoubleData input, GPIndividual gpInd) {
        return getOutputs(state, threadnum, input, gpInd, FeatureCreator.noisyXVals);
    }

    public double[] getOutputs(EvolutionState state, int threadnum, DoubleData input, GPIndividual gpInd, double[] inputs) {
        return getOutputs(state, threadnum, input, gpInd.trees[0], inputs);
    }

    @Override
    public double[][] getAllOutputs(EvolutionState state, int threadnum, GPIndividual gpInd) {
        return new double[][]{getOutputs(state, threadnum, new DoubleData(), gpInd.trees[0], FeatureCreator.noisyXVals)};
    }

    public double[] getOutputs(EvolutionState state, int threadnum, DoubleData input, GPTree tree, double[] inputs) {
        //See FeatureCreator

        double[] results = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            this.x = inputs[i];
            tree.child.eval(state, threadnum, input, stack, tree.owner, this);
            // //TODO: NaN...
            if (Double.isNaN(input.val) || Double.isInfinite(input.val)) {
                //results[i] = xVals[i];
                return null;
            } else {
                //input.val = Math.max(Math.min(1000, input.val), -1000);
                results[i] = input.val;
            }
        }
        DatasetUtils.scaleArray(results);
        if (Double.isNaN(results[0]) || Double.isInfinite(results[0])) {
            return null;
        }
        for (int i = 0; i < results.length; i++) {
            //TODO dodgy?
            results[i] = new BigDecimal(results[i]).setScale(5, BigDecimal.ROUND_HALF_UP).doubleValue();

        }


        return results;
    }

    public double[][] getOutputsForSaving(EvolutionState state, int i, GPIndividual individual) {
        return getAllOutputs(state, i, individual);
    }

    public static class CantComputeMIException extends Exception {
        public CantComputeMIException(String s) {
            super(s);
        }

        public CantComputeMIException() {
            super();
        }
    }
}

