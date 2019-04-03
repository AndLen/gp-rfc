package gp;

import ec.EvolutionState;
import ec.Problem;
import ec.gp.ADFStack;
import ec.gp.GPData;
import ec.gp.GPIndividual;
import ec.gp.GPNode;
import featureCreate.gp.FeatureCreatorProblem;

/**
 * Created by Andrew on 8/04/2015.
 */
public class XNode extends GPNode {

    @Override
    public int expectedChildren() {
        return 0;
    }

    @Override
    public String toString() {
        return "X";
    }

    @Override
    public void eval(EvolutionState state, int thread, GPData input, ADFStack stack, GPIndividual individual, Problem problem) {

        ((DoubleData) input).val = ((FeatureCreatorProblem) problem).x;
    }

}
