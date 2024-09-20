package com.smart3dmap.controller;

import com.smart3dmap.dto.BaseObjectResponse;
import com.smart3dmap.dto.BaseResponse;
import com.smart3dmap.dto.FitRequest;
import org.apache.commons.csv.CSVFormat;
import org.springframework.web.bind.annotation.*;
import smile.base.cart.SplitRule;
import smile.base.mlp.Layer;
import smile.base.mlp.OutputFunction;
import smile.classification.*;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.math.MathEx;
import smile.math.TimeFunction;
import smile.math.distance.Distance;
import smile.math.distance.HammingDistance;
import smile.math.kernel.GaussianKernel;
import smile.validation.CrossValidation;

import java.io.IOException;
import java.net.URISyntaxException;
import java.text.ParseException;

/**
 * 分类算法
 *
 * @author TianSu<ldy_rookie@163.com>
 * @Date 2024/8/29 9:28
 */
@RestController
@RequestMapping("/classification")
public class classificationController {

    /**
     *  K - 最近邻
     * @param request 数据源
     * @return BaseResponse
     */
    @PostMapping("knnFit")
    @ResponseBody
    public BaseResponse knnFit(@RequestBody FitRequest request) throws IOException, ParseException, URISyntaxException {
        DataFrame iris = Read.arff(request.getArffFilePath());
        // 分离特征和标签
        double[][] x = iris.drop("class").toArray();
        int[] y = iris.column("class").toIntArray();
        // 执行交叉验证并返回准确率
        var result = CrossValidation.classification(request.getDatas(), x, y,
                (trainX, trainY) -> KNN.fit(trainX, trainY, request.getK()));
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(result);
        return baseObjectResponse;
    }

    // K - 最近邻距离
    @PostMapping("knnFitDistance")
    @ResponseBody
    public BaseResponse knnFitDistance(@RequestBody FitRequest request) throws IOException, ParseException, URISyntaxException {
        DataFrame iris = Read.arff(request.getArffFilePath());
        // 分离特征和标签
        double[][] x = iris.drop("class").toArray();
        int[] y = iris.column("class").toIntArray();
        //需要自定义距离函数
        Distance distance = new HammingDistance();

        var result = CrossValidation.classification(request.getDatas(), x, y,
                (trainX, trainY) -> KNN.fit(trainX, trainY, request.getK(), distance));
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(result);
        return baseObjectResponse;
    }

    // 隐式决策二维toy数据上 k-NN 的边界
    // K - 最近邻距离
    @PostMapping("kdTree")
    @ResponseBody
    public BaseResponse kdTree(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        DataFrame toy = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter('\t'));
        var x = toy.select(1, 2).toArray();
        var y = toy.column(0).toIntArray();

        var result = KNN.fit(x, y, request.getK());
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(result);
        return baseObjectResponse;
    }

    //---------------------------------------线性判别分析-----------------------------------------------------
    // 线性判别分析----数据源csv？？？？？？？？？？？？？？？？？？？？？
    @PostMapping("ldaFit")
    @ResponseBody
    public BaseResponse ldaFit(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        DataFrame csv = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter('\t'));
        var x = csv.select(1, 2).toArray();
        var y = csv.column(0).toIntArray();
        var fit = LDA.fit(x, y);
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(fit);
        return baseObjectResponse;
    }

    //--------------------------------------Fisher's Linear 判别式-----------------------------------------------
    // 数据源csv？？？？？？？？？？？？？？？？？？？？
    @PostMapping("fldFit")
    @ResponseBody
    public BaseResponse fldFit(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        DataFrame csv = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter('\t'));
        var x = csv.select(1, 2).toArray();
        var y = csv.column(0).toIntArray();
        var fit = FLD.fit(x, y);
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(fit);
        return baseObjectResponse;
    }


    //--------------------------------------正则化判别分析-----------------------------------------------
    // 数据源csv？？？？？？？？？？？？？？？？？？？？
    @PostMapping("rdaFit")
    @ResponseBody
    public BaseResponse rdaFit(@RequestBody FitRequest request) throws IOException, URISyntaxException, ParseException {
        DataFrame iris = Read.arff(request.getArffFilePath());
        // 分离特征和标签
        double[][] x = iris.drop("class").toArray();
        int[] y = iris.column("class").toIntArray();
        // 执行交叉验证并返回准确率
        var result = RDA.fit(x, y, 0.1);
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(result);
        return baseObjectResponse;
    }

    //--------------------------------------Logistic 回归,数据源为txt-----------------------------------------------
    @PostMapping("LogisticRegressionFit")
    @ResponseBody
    public BaseResponse LogisticRegressionFit(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        DataFrame csv = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter('\t'));
        var x = csv.select(1, 2).toArray();
        var y = csv.column(0).toIntArray();
        var fit = LogisticRegression.fit(x, y);
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(fit);
        return baseObjectResponse;
    }

    //--------------------------------------多层感知器神经网络,数据源为txt-----------------------------------------------
    @PostMapping("mlp")
    @ResponseBody
    public BaseResponse mlp(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        DataFrame csv = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter('\t'));
        var x = csv.select(1, 2).toArray();
        var y = csv.column(0).toIntArray();
        var net = new MLP(Layer.input(2),
                Layer.sigmoid(10),
                Layer.mle(1, OutputFunction.SIGMOID)
        );

        net.setLearningRate(TimeFunction.constant(0.1));
        net.setMomentum(TimeFunction.constant(0.1));

        for (int epoch = 0; epoch < 10; epoch++) {
            for (int i : MathEx.permutate(x.length)) {
                net.update(x[i], y[i]);
            }
        }
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(net);
        return baseObjectResponse;
    }

    //--------------------------------------支持向量机,数据源为train-----------------------------------------------
    @PostMapping("svm")
    @ResponseBody
    public BaseResponse svm(@RequestBody FitRequest request) throws IOException, URISyntaxException {
        var zip = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter(' '));
        var x = zip.drop(0).toArray();
        var y = zip.column(0).toIntArray();
        var kernel = new GaussianKernel(8.0);
        var model = OneVersusRest.fit(x, y, (trainX, trainY) -> SVM.fit(trainX, trainY, kernel, 5, 1E-3));
                BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(model);
        return baseObjectResponse;
    }

    //--------------------------------------随机森林,数据源为??-----------------------------------------------
    @PostMapping("randomForest")
    @ResponseBody
    public BaseResponse randomForest(@RequestBody FitRequest request, Formula formula, int ntrees,
                                     int mtry, SplitRule rule, int maxDepth,
                                     int maxNodes, int nodeSize, double subsample) throws IOException, URISyntaxException {
        DataFrame data = Read.csv(request.getArffFilePath(), CSVFormat.DEFAULT.withDelimiter('\t'));
        var model = RandomForest.fit(formula, data, ntrees, mtry, rule, maxDepth, maxNodes, nodeSize, subsample);
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(model);
        return baseObjectResponse;
    }

    //--------------------------------------AdaBoost,数据源为??-----------------------------------------------
    /*@PostMapping("adaBoost")
    @ResponseBody
    public BaseResponse adaBoost(String path) throws IOException {
        List<List<Double>> datas =
                new ArrayList<List<Double>>();
        List<Double> data = new ArrayList<Double>();
        List<Integer> labels = new ArrayList<Integer>();

        String line;
        List<String> lines;
        File file = new File(path);
        BufferedReader reader =
                new BufferedReader(new FileReader(file));
        while ((line = reader.readLine()) != null) {
            lines = Arrays.asList(line.trim().split("\t"));
            for (int i = 0; i < lines.size() - 1; i++) {
                data.add(Double.parseDouble(lines.get(i)));
            }
            labels.add(Integer.parseInt(lines.get(lines.size() - 1)));

            datas.add(data);
            data = new ArrayList<Double>();

        }

        //转换label
        int[] label = new int[labels.size()];
        for (int i = 0; i < label.length; i++) {
            label[i] = labels.get(i);
        }

        //转换属性
        int rows = datas.size();
        int cols = datas.get(0).size();
        double[][] srcData = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                srcData[i][j] = datas.get(i).get(j);
            }
        }

        AdaBoost adaBoost = new AdaBoost(srcData, label, 4, 8);
        double right = 0;
        for (int i = 0; i < srcData.length; i++) {
            int tag = adaBoost.predict(srcData[i]);
            if (i % 10 == 0) System.out.println();
            System.out.print(tag + " ");
            if (tag == label[i]) {
                right += 1;
            }
        }
        right = right / srcData.length;
        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(right * 100);
        return baseObjectResponse;
    }*/

}
