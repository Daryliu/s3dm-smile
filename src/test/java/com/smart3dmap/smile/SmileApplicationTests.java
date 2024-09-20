package com.smart3dmap.smile;

import com.smart3dmap.dto.BaseObjectResponse;
import org.apache.commons.csv.CSVFormat;
import smile.classification.KNN;
import smile.classification.LDA;
import smile.data.DataFrame;
import smile.io.Read;
import smile.validation.CrossValidation;

import java.io.IOException;
import java.net.URISyntaxException;
import java.text.ParseException;

class SmileApplicationTests {

    public static void main(String[] args) throws IOException, URISyntaxException {
        DataFrame toy = Read.csv("E:/00-Code/05-Temp/smile-3.1.1/data/classification/toy200.txt",
                CSVFormat.DEFAULT.withDelimiter('\t'));
        var x = toy.select(1, 2).toArray();
        var y = toy.column(0).toIntArray();
        System.out.println(LDA.fit(x, y));

    }

    void contextLoads() throws IOException, ParseException, URISyntaxException {
                var iris = Read.arff("E:/00-Code/05-Temp/smile-3.1.1/data/weka/iris.arff");
        var x = iris.drop("class").toArray();
        var y = iris.column("class").toIntArray();
        CrossValidation.classification(10, x, y, (trainX, trainY) -> KNN.fit(trainX, trainY, 3));
//        var canvas = ScatterPlot.of(iris, "sepallength", "sepalwidth", "class", '*').canvas();
//        canvas.setAxisLabels("sepallength", "sepalwidth");
//        canvas.window();

        /*var canvas = ScatterPlot.of(iris, "sepallength", "sepalwidth", "petallength", "class", '*').canvas();
        canvas.setAxisLabels("sepallength", "sepalwidth", "petallength");
        canvas.window();*/

        /*var canvas = PlotGrid.splom(iris, '*', "class");
        canvas.window();*/

        /*double[][] heart = new double[200][2];
        for (int i = 0; i < 200; i++) {
            double t = PI * (i - 100) / 100;
            heart[i][0] = 16 * pow(sin(t), 3);
            heart[i][1] = 13 * cos(t) - 5 * cos(2*t) - 2 * cos(3*t) - cos(4*t);
        }
        var canvas = LinePlot.of(heart, Color.RED).canvas();
        canvas.window();*/

        // the matrix to display
        /*double[][] z = {
                {1.0, 2.0, 4.0, 1.0},
                {6.0, 3.0, 5.0, 2.0},
                {4.0, 2.0, 1.0, 5.0},
                {5.0, 4.0, 2.0, 3.0}
        };

        // make the matrix larger with bicubic interpolation
        double[] x = {0.0, 1.0, 2.0, 3.0};
        double[] y = {0.0, 1.0, 2.0, 3.0};
        var bicubic = new BicubicInterpolation(x, y, z);
        var Z = new double[101][101];
        for (int i = 0; i <= 100; i++) {
            for (int j = 0; j <= 100; j++)
                Z[i][j] = bicubic.interpolate(i * 0.03, j * 0.03);
        }

        Heatmap.of(Z, Palette.jet(256)).canvas().window();*/
    }

}
