package com.smart3dmap.controller;

import com.smart3dmap.dto.BaseObjectResponse;
import com.smart3dmap.dto.BaseResponse;
import com.smart3dmap.dto.FitRequest;
import org.apache.commons.csv.CSVFormat;
import org.springframework.web.bind.annotation.*;
import smile.base.mlp.Layer;
import smile.base.mlp.OutputFunction;
import smile.classification.KNN;
import smile.classification.LDA;
import smile.classification.MLP;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.feature.extraction.PCA;
import smile.feature.selection.GAFE;
import smile.feature.selection.SumSquaresRatio;
import smile.feature.transform.Standardizer;
import smile.feature.transform.WinsorScaler;
import smile.gap.BitString;
import smile.io.Read;
import smile.math.MathEx;
import smile.math.TimeFunction;
import smile.plot.swing.ScatterPlot;
import smile.plot.swing.ScreePlot;
import smile.validation.CrossValidation;
import smile.validation.metric.Accuracy;

import java.io.IOException;
import java.net.URISyntaxException;
import java.text.ParseException;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Feature Engineering算法
 *
 * @author TianSu<ldy_rookie@163.com>
 * @Date 2024/8/31 16:57
 */
@RestController
@RequestMapping("/featureEngineer")
public class FeatureEngineeringController {

    //-------------------------------Preprocessing,data/classification/pendigits.txt或data/usps/zip.train------------------------------------
    @PostMapping("preprocess")
    @ResponseBody
    public BaseResponse preprocess(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        AtomicReference<MLP> result = new AtomicReference<>();
        if (request.getArffFilePath().endsWith(".txt")) {
            var pendigits = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter('\t'));
            var df = pendigits.drop(16);
            var y = pendigits.column(16).toIntArray();
            var scaler = WinsorScaler.fit(df, 0.01, 0.99);
            var x = scaler.apply(df).toArray();
            CrossValidation.classification(10, x, y, (trainX, trainY) -> {
                var model = new smile.classification.MLP(Layer.input(16),
                        Layer.sigmoid(50),
                        Layer.mle(10, OutputFunction.SIGMOID)
                );

                for (int epoch = 0; epoch < 10; epoch++) {
                    for (int i : MathEx.permutate(x.length)) {
                        model.update(x[i], y[i]);
                    }
                }
                result.set(model);
                return model;
            });
        } else if(request.getArffFilePath().endsWith(".train")) {
            var zip = Read.csv("data/usps/zip.train", CSVFormat.DEFAULT.withDelimiter(' '));
            var df = zip.drop(0);
            var y = zip.column(0).toIntArray();

            var scaler = Standardizer.fit(df);
            var x = scaler.apply(df).toArray();

            var model = new smile.classification.MLP(Layer.input(256),
                    Layer.sigmoid(768),
                    Layer.sigmoid(192),
                    Layer.sigmoid(30),
                    Layer.mle(10, OutputFunction.SIGMOID)
            );

            model.setLearningRate(TimeFunction.constant(0.1));
            model.setMomentum(TimeFunction.constant(0.0));

            for (int epoch = 0; epoch < 10; epoch++) {
                for (int i : MathEx.permutate(x.length)) {
                    model.update(x[i], y[i]);
                }
            }
            result.set(model);
        } else {
            System.out.println("文件格式错误");
        }

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(result);
        return baseObjectResponse;
    }

    //-------------------------------Sum Squares Ratio,data/weka/iris.arff------------------------------------
    @PostMapping("sumSquaresRatio")
    @ResponseBody
    public BaseResponse sumSquaresRatio(@RequestBody FitRequest request) throws IOException, ParseException, URISyntaxException {
        DataFrame iris = Read.arff(request.getArffFilePath());
        var model = SumSquaresRatio.fit(iris, "class");
        // 散点图
//        var df = iris.select("petallength", "petalwidth", "class");
//        var canvas = ScatterPlot.of(df, "petallength", "petalwidth", "class", '*').canvas();
//        canvas.setAxisLabels(iris.names()[2], iris.names()[3]);
//        canvas.window();
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(model);
        return baseObjectResponse;
    }

    //-------------------------------Ensemble Learning Based Feature Selection,data/weka/iris.arff------------------------------------
    @PostMapping("ensembleLearning")
    @ResponseBody
    public BaseResponse ensembleLearning(@RequestBody FitRequest request) throws IOException, ParseException, URISyntaxException {
        DataFrame iris = Read.arff(request.getArffFilePath());
        var model = smile.classification.RandomForest.fit(Formula.lhs("class"), iris);
        /*for (int i = 0; i < 4; i++) {
            System.out.format("%s\t%.2f\n", iris.names()[i], model.importance()[i]);
        }*/
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(model);
        return baseObjectResponse;
    }

    //---------------------------------------------TreeSHAP,data/weka/regression/housing.arff--------------------------------
    @PostMapping("treeSHAP")
    @ResponseBody
    public BaseResponse treeSHAP(@RequestBody FitRequest request) throws IOException, ParseException, URISyntaxException {
        DataFrame housing = Read.arff(request.getArffFilePath());
        var model = smile.regression.RandomForest.fit(Formula.lhs("class"), housing);
        var importance = model.importance();
        var shap = model.shap(housing.stream().parallel());
        var fields = java.util.Arrays.copyOf(housing.names(), 13);
        smile.sort.QuickSort.sort(importance, fields);
        smile.sort.QuickSort.sort(shap, fields);
//        for (int i = 0; i < importance.length; i++) {
//            System.out.format("%-15s %12.4f%n", fields[i], importance[i]);
//        }
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(model);
        return baseObjectResponse;
    }

    //---------------Genetic Algorithm Based Feature Selection,data/usps/zip.train及data/usps/zip.test--------------------------------
    @PostMapping("geneticAlgorithm")
    @ResponseBody
    public BaseResponse geneticAlgorithm(@RequestBody FitRequest request) throws IOException, ParseException, URISyntaxException {
        var train = Read.csv("data/usps/zip.train", CSVFormat.DEFAULT.withDelimiter(' '));
        var test = Read.csv("data/usps/zip.test", CSVFormat.DEFAULT.withDelimiter(' '));
        var x = train.drop("V1").toArray();
        var y = train.column("V1").toIntArray();
        var testx = test.drop("V1").toArray();
        var testy = test.column("V1").toIntArray();

        var selection = new GAFE();
        var result = selection.apply(100, 20, 256,
                GAFE.fitness(x, y, testx, testy, new Accuracy(),
                        (trainX, trainY) -> LDA.fit(trainX, trainY)));

//        for (BitString bits : result) {
//            System.out.format("%.2f%% %s%n", 100*bits.fitness(), bits);
//        }
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(result);
        return baseObjectResponse;
    }

    //--------------------Principal Component Analysis,data/classification/pendigits.txt-------------------------------
    @PostMapping("pca")
    @ResponseBody
    public BaseResponse pca(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var pendigits = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter('\t'));
        var formula = Formula.lhs("V17");
        var x = formula.x(pendigits).toArray();
        var y = formula.y(pendigits).toIntArray();

        var pca = PCA.fit(x);
//        var canvas = new ScreePlot(pca.varianceProportion()).canvas();
//        canvas.window();
//
//        var p = pca.getProjection(3);
//        var x2 = p.apply(x);
//        ScatterPlot.of(x2, y, '.').canvas().window();
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(pca);
        return baseObjectResponse;
    }
}
