package featureCreate.gp;

import ec.EvolutionState;
import ec.gp.GPIndividual;
import ec.simple.SimpleProblemForm;

/**
 * Created by lensenandr on 29/11/17.
 */
public interface FCProblemInterface extends SimpleProblemForm {
    double MIN_SOURCE_MI = 0.70;
    boolean CAP_SOURCE_MI = false;

    double[][] getAllOutputs(EvolutionState state, int threadnum, GPIndividual gpInd);
}
