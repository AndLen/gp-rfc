package gp;

/**
 * Created by Andrew on 8/04/2015.
 */
public class Mul extends AbstractArithmeticNode {
    @Override
    public String toString() {
        return "mul";
    }

    @Override
    protected double performOperation(double child1, double child2) {
        return child1 * child2;
    }
}
