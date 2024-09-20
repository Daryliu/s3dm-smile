package com.smart3dmap.controller;

import com.smart3dmap.dto.BaseObjectResponse;
import com.smart3dmap.dto.BaseResponse;
import com.smart3dmap.dto.FitRequest;
import org.apache.commons.csv.CSVFormat;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.ResponseBody;
import smile.classification.DecisionTree;
import smile.classification.LogisticRegression;
import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.validation.CrossValidation;

import java.io.IOException;
import java.net.URISyntaxException;
import java.text.ParseException;

/**
 * 模型验证算法
 *
 * @author TianSu<ldy_rookie@163.com>
 * @Date 2024/8/31 17:52
 */
public class ModelValidationController {

    //---------------------------------------Evaluation Metrics,-----------------------------------------
    @PostMapping("evaluationMetrics")
    @ResponseBody
    public BaseResponse evaluationMetrics(@RequestBody FitRequest request) throws IOException, ParseException, URISyntaxException {
//        DataFrame iris = Read.arff(request.getArffFilePath());
        LogisticRegression result = null;
        RandomForest modelArff;
        if (request.getArffFilePath().endsWith(".arff")) {
            var segTrain = Read.arff("data/weka/segment-challenge.arff");
            var segTest = Read.arff("data/weka/segment-test.arff");

            var model = RandomForest.fit(Formula.lhs("class"), segTrain);
            modelArff = model;
//            var pred = model.predict(segTest);
//            var result = Accuracy.of(segTest.column("class").toIntArray(), pred);
        } else if (request.getArffFilePath().endsWith(".txt")) {
            var toyTrain = Read.csv("data/classification/toy200.txt", CSVFormat.DEFAULT.withDelimiter('\t'));
            var toyTest = Read.csv("data/classification/toy20000.txt", CSVFormat.DEFAULT.withDelimiter('\t'));

            var x = toyTrain.select(1, 2).toArray();
            var y = toyTrain.column(0).toIntArray();
            var model = LogisticRegression.fit(x, y, 0.1, 0.001, 100);
            result = model;

//            var testx = toyTest.select(1, 2).toArray();
//            var testy = toyTest.column(0).toIntArray();
//            var pred = Arrays.stream(testx).mapToInt(xi -> model.predict(xi)).toArray();
//            Accuracy.of(testy, pred);
//            Recall.of(testy, pred);
//            Sensitivity.of(testy, pred);
//            Specificity.of(testy, pred);
//            Fallout.of(testy, pred);
//            FDR.of(testy, pred);
//            FScore.F1.score(testy, pred);
        }

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(result);
        return baseObjectResponse;
    }

    //---------------------------------------Cross Validation,data/weka/iris.arff-----------------------------------------
    @PostMapping("crossValidation")
    @ResponseBody
    public BaseResponse crossValidation(@RequestBody FitRequest request) throws IOException, ParseException, URISyntaxException {
        DataFrame iris = Read.arff(request.getArffFilePath());
        var result = CrossValidation.classification(10, Formula.lhs("class"), iris,
                (formula, data) -> DecisionTree.fit(formula, data));
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(result);
        return baseObjectResponse;
    }


    //---------------------------------------Cross -------------------------------------------------------------------
    /*@PostMapping("cross")
    @ResponseBody
    public BaseResponse cross(@RequestBody FitRequest request) throws IOException, ParseException, URISyntaxException {
        DataFrame iris = Read.arff(request.getArffFilePath());
        var result = CrossValidation.classification(10, Formula.lhs("class"), iris,
                (formula, data) -> DecisionTree.fit(formula, data));
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(result);
        return baseObjectResponse;
    }*/

}
