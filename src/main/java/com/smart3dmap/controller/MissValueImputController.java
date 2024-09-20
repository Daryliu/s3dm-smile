package com.smart3dmap.controller;

import com.smart3dmap.dto.BaseObjectResponse;
import com.smart3dmap.dto.BaseResponse;
import com.smart3dmap.dto.FitRequest;
import org.apache.commons.csv.CSVFormat;
import org.springframework.web.bind.annotation.*;
import smile.data.Tuple;
import smile.feature.imputation.KMedoidsImputer;
import smile.feature.imputation.KNNImputer;
import smile.feature.imputation.SVDImputer;
import smile.feature.imputation.SimpleImputer;
import smile.io.Read;
import smile.math.MathEx;
import smile.math.distance.Distance;

import java.io.IOException;
import java.net.URISyntaxException;

/**
 * Missing Value Imputation
 *
 * @author TianSu<ldy_rookie@163.com>
 * @Date 2024/9/11 17:45
 */
@RestController
@RequestMapping("/missValueImput")
public class MissValueImputController {

    //---------------------------SimpleImputer,data/clustering/synthetic_control.data---------------------------------
    @PostMapping("simpleImputer")
    @ResponseBody
    public BaseResponse simpleImputer(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var format = CSVFormat.Builder.create().setDelimiter(' ').build();
        var data = Read.csv(request.getArffFilePath(), format);
        SimpleImputer imputer = SimpleImputer.fit(data);
        var completeData = imputer.apply(data);

//        DataFrame iris = Read.arff(request.getArffFilePath());
//        // 分离特征和标签
//        double[][] x = iris.drop("class").toArray();
//        int[] y = iris.column("class").toIntArray();
//        // 执行交叉验证并返回准确率
//        var result = CrossValidation.classification(request.getDatas(), x, y,
//                (trainX, trainY) -> KNN.fit(trainX, trainY, request.getK()));
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(completeData);
        return baseObjectResponse;
    }

    //---------------------------KNNImputer,data/clustering/synthetic_control.data---------------------------------
    @PostMapping("knnImputer")
    @ResponseBody
    public BaseResponse knnImputer(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var format = CSVFormat.Builder.create().setDelimiter(' ').build();
        var data = Read.csv(request.getArffFilePath(), format);
        var imputer = new KNNImputer(data, 5);
        var completeData = imputer.apply(data);

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(completeData);
        return baseObjectResponse;
    }

    //---------------------------KMedoidsImputer,data/clustering/synthetic_control.data---------------------------------
    @PostMapping("kMedoidsImputer")
    @ResponseBody
    public BaseResponse kMedoidsImputer(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var format = CSVFormat.Builder.create().setDelimiter(' ').build();
        var data = Read.csv(request.getArffFilePath(), format);
        Distance<Tuple> distance = (x, y) -> {
            double[] xd = x.toArray();
            double[] yd = y.toArray();
            return MathEx.squaredDistanceWithMissingValues(xd, yd);
        };
        var imputer = KMedoidsImputer.fit(data, distance,20);
        var completeData = imputer.apply(data);

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(completeData);
        return baseObjectResponse;
    }

    //---------------------------SVD 插补,data/clustering/synthetic_control.data---------------------------------
    @PostMapping("svdImputation")
    @ResponseBody
    public BaseResponse svdImputation(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var format = CSVFormat.Builder.create().setDelimiter(' ').build();
        var data = Read.csv(request.getArffFilePath(), format);
        var matrix = data.toArray();
        double[][] completeMatrix = SVDImputer.impute(matrix, 5, 10);

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(completeMatrix);
        return baseObjectResponse;
    }

}
