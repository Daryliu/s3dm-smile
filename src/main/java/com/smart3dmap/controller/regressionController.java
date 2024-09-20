package com.smart3dmap.controller;

import com.smart3dmap.dto.BaseObjectResponse;
import com.smart3dmap.dto.BaseResponse;
import com.smart3dmap.dto.FitRequest;
import org.apache.commons.csv.CSVFormat;
import org.springframework.web.bind.annotation.*;
import smile.base.rbf.RBF;
import smile.clustering.KMeans;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.math.kernel.GaussianKernel;
import smile.regression.*;
import smile.validation.CrossValidation;

import java.io.IOException;
import java.net.URISyntaxException;
import java.text.ParseException;

/**
 * 回归算法
 *
 * @author TianSu<ldy_rookie@163.com>
 * @Date 2024/8/29 10:02
 */
@RestController
@RequestMapping("/regression")
public class regressionController {

    //------------------------------------------普通最小二乘法,data/weka/regression/2dplanes.arff----------------------------------------------------------
    @PostMapping("ols")
    @ResponseBody
    public BaseResponse ols(@RequestBody FitRequest request) throws IOException, ParseException, URISyntaxException {
        DataFrame iris = Read.arff(request.getArffFilePath());
        var model = OLS.fit(Formula.lhs("y"), iris);
//        model.predict(iris.get(0));
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(model);
        return baseObjectResponse;
    }

    //----------------------------------------Ridge Regression，data/weka/regression/longley.arff------------------------------------------------------------
    @PostMapping("ridgeRegression")
    @ResponseBody
    public BaseResponse ridgeRegression(@RequestBody FitRequest request) throws IOException, ParseException, URISyntaxException {
        DataFrame iris = Read.arff(request.getArffFilePath());
        var model = RidgeRegression.fit(Formula.lhs("employed"), iris,0.0057);
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(model);
        return baseObjectResponse;
    }

    //----------------------------------------Lasso Regression，data/regression/diabetes.csv------------------------------------------------------------
    @PostMapping("lassoRegression")
    @ResponseBody
    public BaseResponse lassoRegression(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        DataFrame csv = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withFirstRecordAsHeader());
        var formula = Formula.lhs("y");
        var model = LASSO.fit(formula,csv, request.getK());
//        LASSO.fit(formula,csv, 100);
//        LASSO.fit(formula,csv, 500);
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(model);
        return baseObjectResponse;
    }

    //----------------------------------------Radial Basis Function Networks，------------------------------------------------------------
    @PostMapping("rbfNetwork")
    @ResponseBody
    public BaseResponse rbfNetwork(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        DataFrame diabetes = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withFirstRecordAsHeader());
        var y = diabetes.column("y").toDoubleArray();
        var x = diabetes.select(1, 2, 3, 4, 5, 6, 7, 8, 9, 10).toArray(); // use only the primary attributes
        var model = CrossValidation.regression(10, x, y, (trainX, trainY) -> smile.regression.RBFNetwork.fit(trainX, trainY,
                RBF.fit(trainX, 10), false));
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(model);
        return baseObjectResponse;
    }

    //----------------------------------------Support Vector Regression，------------------------------------------------------------
    @PostMapping("svm")
    @ResponseBody
    public BaseResponse svm(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        DataFrame diabetes = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withFirstRecordAsHeader());
        var y = diabetes.column("y").toDoubleArray();
        var x = diabetes.select(1, 2, 3, 4, 5, 6, 7, 8, 9, 10).toArray();
        var model = CrossValidation.regression(10, x, y, (trainX, trainY) -> smile.regression.SVM.fit(trainX, trainY,
                new GaussianKernel(0.06), 20, 10, 1E-3));
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(model);
        return baseObjectResponse;
    }

    //----------------------------------------Regression Tree，------------------------------------------------------------
    @PostMapping("regressionTree")
    @ResponseBody
    public BaseResponse regressionTree(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        DataFrame diabetes = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withFirstRecordAsHeader());
        //?????formula, data
        var model = CrossValidation.regression(10, Formula.lhs("y"), diabetes,
                (formula, data) -> RegressionTree.fit(formula, data));
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(model);
        return baseObjectResponse;
    }

    //----------------------------------------Random Forest，------------------------------------------------------------
    @PostMapping("randomForest")
    @ResponseBody
    public BaseResponse randomForest(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        DataFrame diabetes = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withFirstRecordAsHeader());
        //?????formula, data
        var model = CrossValidation.regression(10, Formula.lhs("y"), diabetes,
                (formula, data) -> RandomForest.fit(formula, data));
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(model);
        return baseObjectResponse;
    }

    //----------------------------------------Gradient Boosting，------------------------------------------------------------
    @PostMapping("gradientTreeBoost")
    @ResponseBody
    public BaseResponse gradientTreeBoost(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        DataFrame diabetes = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withFirstRecordAsHeader());
        //?????formula, data
        var model = CrossValidation.regression(10, Formula.lhs("y"), diabetes,
                (formula, data) -> smile.regression.GradientTreeBoost.fit(formula, data));
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(model);
        return baseObjectResponse;
    }

    //----------------------------------------Gaussian Process，------------------------------------------------------------
    @PostMapping("gaussianProcessRegression")
    @ResponseBody
    public BaseResponse gaussianProcessRegression(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        DataFrame diabetes = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withFirstRecordAsHeader());
        var y = diabetes.column("y").toDoubleArray();
        var x = diabetes.select(1, 2, 3, 4, 5, 6, 7, 8, 9, 10).toArray();
        var model = CrossValidation.regression(10, x, y, (trainX, trainY) -> {
            var t = KMeans.fit(x, 20).centroids;
            return GaussianProcessRegression.fit(x, y, t, new GaussianKernel(0.06), 0.01);
        });
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(model);
        return baseObjectResponse;
    }
}
