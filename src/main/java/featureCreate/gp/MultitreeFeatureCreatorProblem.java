package featureCreate.gp;

import ec.EvolutionState;
import ec.Individual;
import ec.gp.GPIndividual;
import ec.gp.GPTree;
import ec.simple.SimpleFitness;
import featureCreate.FeatureCreator;
import gp.DoubleData;


/**
 * Created by lensenandr on 8/08/16.
 */
public class MultitreeFeatureCreatorProblem extends FeatureCreatorProblem {

    @Override
    public void evaluate(EvolutionState state, Individual ind, int subpopulation, int threadnum) {
        if (!ind.evaluated)  // don't bother reevaluating
        {
            GPIndividual gpInd = (GPIndividual) ind;
            GPTree[] trees = gpInd.trees;

            double[][] outputs = getAllOutputs(state, threadnum, gpInd);

            double fitness = internalMeasureFitness(outputs, trees);

            SimpleFitness f = ((SimpleFitness) ind.fitness);
            f.setFitness(state, fitness, false);

            ind.evaluated = true;

        }

    }

    public double[][] getAllOutputs(EvolutionState state, int threadnum, GPIndividual gpInd) {
        GPTree[] trees = gpInd.trees;
        double[][] outputs = new double[trees.length][FeatureCreator.noisyXVals.length];

        for (int i = 0; i < trees.length; i++) {
            outputs[i] = getOutputs(state, threadnum, new DoubleData(), trees[i], FeatureCreator.noisyXVals);
        }
        return outputs;
    }


}


