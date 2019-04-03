package featureCreate.gp;

import ec.EvolutionState;
import ec.gp.GPIndividual;
import ec.gp.GPTree;
import featureCreate.FeatureCreator;
import featureGrouping.MutualInformationMap;
import gp.DoubleData;
import other.DatasetUtils;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.OptionalDouble;


/**
 * Created by lensenandr on 8/08/16.
 */
public class MTFCMultivariateProblem extends MTFCTreeSimilarityProblem {
    public double[] xVals;

    static double getMutualInfo(double[][] outputs) {
        return MutualInformationMap.getMultiVarMutualInformation(FeatureCreator.multipleXVals, outputs, false) / FeatureCreator.baseMI;
    }

    public double[][] getAllOutputs(EvolutionState state, int threadnum, GPIndividual gpInd) {
        GPTree[] trees = gpInd.trees;
        double[][] outputs = new double[trees.length][];

        for (int i = 0; i < trees.length; i++) {
            outputs[i] = getOutputs(state, threadnum, new DoubleData(), trees[i], FeatureCreator.multipleNoisyXVals);
        }
        return outputs;

    }

    public double[] getOutputs(EvolutionState state, int threadnum, DoubleData input, GPTree tree, double[][] multipleXVals) {
        int numSources = multipleXVals.length;
        int numInputs = multipleXVals[0].length;
        double[] results = new double[numInputs];
        for (int i = 0; i < numInputs; i++) {
            this.xVals = new double[numSources];
            for (int j = 0; j < numSources; j++) {
                //TODO: Careful....
                this.xVals[j] = multipleXVals[j][i];
            }
            tree.child.eval(state, threadnum, input, stack, tree.owner, this);
            // //TODO: NaN...
            if (Double.isNaN(input.val) || Double.isInfinite(input.val)) {
                return null;
            } else {
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

    @Override
    public double internalMeasureFitness(double[][] outputs, GPTree[] trees) {
        double fitness;

        for (int i = 0; i < outputs.length; i++) {
            if (outputs[i] == null) {
                //Okay, so not a valid solution. fitness will be -ve numer of invalid trees.
                int count = 0;
                for (int j = i; j < outputs.length; j++) {
                    if (outputs[j] == null) {
                        count++;
                    }
                }
                return -count;

                //throw new CantComputeMIException();
            }
        }

        double worstSourceMI = getMutualInfo(outputs);

        if (CAP_SOURCE_MI) {
            worstSourceMI = Math.min(worstSourceMI, 0.9);
        }
        double worstDifference = 0;
        if (outputs.length > 1) {
            double[] minDifferences = getMinDistBetweenOutputs(outputs);
            OptionalDouble worst = Arrays.stream(minDifferences).min();//.getAsDouble();
            if (worst.isPresent() && !Double.isNaN(worst.getAsDouble())) {
                worstDifference = worst.getAsDouble();
            }
        } else worstDifference = 1;
        fitness = ((worstSourceMI) * worstDifference);//+0.1*worstSourceMI;


        if (Double.isNaN(fitness)) {
            //System.err.println();
        }
        return fitness;
    }

}


