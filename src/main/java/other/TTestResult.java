package other;

public enum TTestResult {
    NO_SIGNIFICANCE, BETTER, WORSE;

    public String toHuman() {
        switch (this) {
            case BETTER:
                return "(+)";
            case WORSE:
                return "(-)";
            case NO_SIGNIFICANCE:
                return "   ";
            default:
                throw new IllegalStateException();
        }
    }
}
