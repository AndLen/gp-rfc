package other;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Properties;

import static other.Util.IN_A_JAR;

/**
 * Created by lensenandr on 17/03/16.
 */
public class BetterProperties extends Properties {

    public int getInt(String key) {
        return Integer.parseInt(getProperty(key));
    }

    public boolean getBoolean(String key) {
        return Boolean.parseBoolean(getProperty(key));
    }

    public static BufferedReader readContextually(String configFilePath) {
        BufferedReader bufferedReader;
        if (IN_A_JAR) {
            System.out.println("Inside a jar, loading config internally.");
            InputStream resource = BetterProperties.class.getClassLoader().getResourceAsStream(configFilePath.split("/config/")[1]);
            System.out.println(resource);
            bufferedReader = new BufferedReader(new InputStreamReader(resource));
        } else {
            try {
                bufferedReader = Files.newBufferedReader(Paths.get(configFilePath));
            } catch (IOException e) {
                throw new Error(e);
            }
        }
        return bufferedReader;
    }

    public void build(String[] args) throws IOException {
        IN_A_JAR = BetterProperties.class.getResource("/other/BetterProperties.class").getProtocol().equals("jar");
        BufferedReader bufferedReader = readContextually(args[0]);

        this.putAll(internalLoad(bufferedReader, Paths.get(args[0]).getParent()));
        for (int i = 1; i < args.length; i++) {
            String[] split = args[i].split("=");
            if (split.length == 2) {
                put(split[0].trim(), split[1].trim());
            }
        }
    }

    private BetterProperties internalLoad(BufferedReader reader, Path directory) throws IOException {
        BetterProperties properties = new BetterProperties();
        properties.load(reader);
        if (properties.containsKey("parent")) {
            //load parent first

            Path parentFile = directory.resolve(properties.getProperty("parent"));
            BufferedReader parentReader = readContextually(parentFile.toString());
            BetterProperties parentProperties = internalLoad(parentReader, parentFile.getParent());
            //Overwrite parent with our child properties
            parentProperties.putAll(properties);
            return parentProperties;
        } else {
            return properties;
        }
    }

    public double getDouble(String key) {
        return Double.parseDouble(getProperty(key));
    }
}
