# chatbot-picture
This model was used for reply gif files in wechat. It works as follows: 

 1.crawl an large amounts of expression packs form the internet.
 
 2.extract the features from the crawled expressionss with the help of trained vgg19. 
   we can generate an array which shape is (num_expressions, num_features)
 
 3.for any input image and gif, also extract features and get the most similar 20 expressions by using Nearest neighboors
 
 To start your model, run:
 
```python
git clone git@github.com:DeeChat/chatbot-picture.git
cd chatbot-picture
python expession_pack.py
```
 onece crawl all expresssions
 
 ```python
 python vgg19.py
 ```
 after several hours, we can get features_mat.npy, which store the all images' feature
 
 ```python
 python wechat.py
 ```
 we can deploy chatbot-picture on wechat
