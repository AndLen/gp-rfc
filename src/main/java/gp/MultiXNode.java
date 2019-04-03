package gp;

import ec.EvolutionState;
import ec.Problem;
import ec.gp.*;
import featureCreate.FeatureCreator;
import featureCreate.gp.MTFCMultivariateProblem;

/**
 * Created by Andrew on 8/04/2015.
 */
public class MultiXNode extends ERC {

    int val;

    public int getVal() {
        return val;
    }

    @Override
    public String toString() {
        return "X" + val;
    }

    @Override
    public String encode() {
        return toString();
    }

    @Override
    public int expectedChildren() {
        return 0;
    }

    @Override
    public void resetNode(EvolutionState state, int thread) {
        val = state.random[thread].nextInt(FeatureCreator.NUM_SOURCE_MV);
    }


    @Override
    public boolean nodeEquals(GPNode node) {
        if (node instanceof MultiXNode) {
            MultiXNode multiXNode = (MultiXNode) node;
            return multiXNode.val == val;
        } else return false;
    }

    @Override
    public void eval(EvolutionState state, int thread, GPData input, ADFStack stack, GPIndividual individual, Problem problem) {
        ((DoubleData) input).val = ((MTFCMultivariateProblem) problem).xVals[this.val];
    }

    public int nodeHashCode() {
        // a reasonable hash code
        return this.getClass().hashCode() + val;
    }
    //    public GPNode clone(){

}
