package com.smart3dmap.controller;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * NLP
 *
 * @author TianSu<ldy_rookie@163.com>
 * @Date 2024/9/12 14:37
 */
@RestController
@RequestMapping("/nlp")
public class NlpController {

    //Bag of Words,data/text/movie.txt
    /*@PostMapping("normalization")
    @ResponseBody
    public BaseResponse normalization(@RequestBody FitRequest request) throws IOException {
        var lines = Files.lines(java.nio.file.Paths.get(request.getArffFilePath()));
        var corpus = lines.map(line -> {
            var sentences = SimpleSentenceSplitter.getInstance().split(SimpleNormalizer.getInstance().normalize(line));
            var words = Arrays.stream(sentences).
                    flatMap(s -> Arrays.stream(tokenizer.split(s))).
                    filter(w -> !(EnglishStopWords.DEFAULT.contains(w.toLowerCase()) || EnglishPunctuations.getInstance().contains(w))).
                    toArray(String[]::new);

            var bag = Arrays.stream(words).
                    map(porter::stem).
                    map(String::toLowerCase).
                    collect(Collectors.groupingBy(java.util.function.Function.identity(), Collectors.summingInt(e -> 1)));

            return bag;
        });

        String[] features = {"like", "good", "perform", "littl", "love", "bad", "best"};
        var zero = Integer.valueOf(0);
        var bags = corpus.map(bag -> {
            double[] x = new double[features.length];
            for (int i = 0; i < x.length; i++) x[i] = (Integer) bag.getOrDefault(features[i], zero);
            return x;
        }).toArray(double[][]::new);

        var n = bags.length;
        int[] df = new int[features.length];
        for (double[] bag : bags) {
            for (int i = 0; i < df.length; i++) {
                if (bag[i] > 0) df[i]++;
            }
        }

        var data = Arrays.stream(bags).map(bag -> {
            var maxtf = MathEx.max(bag);
            double[] x = new double[bag.length];

            for (int i = 0; i < x.length; i++) {
                x[i] = (bag[i] / maxtf) * Math.log((1.0 + n) / (1.0 + df[i]));
            }

            MathEx.unitize(x);
            return x;
        }).toArray(double[][]::new);

        BaseObjectResponse baseObjectResponse = new BaseObjectResponse();
        baseObjectResponse.setData(data);
        return baseObjectResponse;
    }*/

}
