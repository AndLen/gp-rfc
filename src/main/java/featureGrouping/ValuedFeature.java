package featureGrouping;

/**
 * Created by lensenandr on 9/08/17.
 */
public class ValuedFeature {
    public final double[] values;
    public final int featureID;

    public ValuedFeature(double[] values, int featureID) {
        this.values = values;
        this.featureID = featureID;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        ValuedFeature that = (ValuedFeature) o;

        return featureID == that.featureID;

    }

    @Override
    public int hashCode() {
        return featureID;
    }
}
