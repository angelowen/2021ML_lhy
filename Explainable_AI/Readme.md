## Pretrain model
[link](https://drive.google.com/file/d/1CShZHsO8oAZwxQkMe7jRtEgSNb2w_OZu/edit)

## Lime 
一個關於解釋機器學習分類器正在做什麼的package

## Saliency map
Highlight 在分類任務中貢獻最大的輸入圖像的像素的Heatmaps
與loss相關:
* image
* model parameters
* label
當我們改變圖像的像素值時，loss對圖像的偏微分值顯示了loss的變化。 我們可以說它意味著像素的重要性。 我們可以將其可視化以展示圖像的哪一部分對模型的判斷貢獻最大。
## Smooth grad
Smooth grad 的方法是，在圖片中隨機地加入 noise，然後得到不同的 heatmap，把這些 heatmap 平均起來就得到一個比較能抵抗 noisy gradient 的結果。

## Filter explanation
可畫出模型第n層輸出的可解釋性
這裡我們想要知道某一個 filter 到底認出了什麼。我們會做以下兩件事情：
* Filter activation: 挑幾張圖片出來，看看圖片中哪些位置會 activate 該 filter
* Filter visualization: 怎樣的 image 可以最大程度的 activate 該 filter

# Integrated Gradient
鼻子長度和分類為大象機率圖所示，當鼻子長度小於1.0米時，可求得梯度不為0，但當鼻子長度接近或大於1.0米時，則是判定為大象的機率已接近百分之百（1.0），此時不管鼻長再加長，是大象的機率也不會再高了，如此就會造成計算出的梯度反而變為0（梯度飽和）的現象產生。
為解決這項問題，提出不直接以梯度當作特徵重要性，而是將所有梯度積分（加總）起來，再其來表示重要性，如此就不會有梯度為0（飽和）的問題產生