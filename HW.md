### HW1

- 改善网络架构

  - BatchNorm
  - Dropout
  - 网络加深
  - 更换激活函数

- 做Feature Selection，尝试使用chi2、f_regression

  ```python
  from sklearn.feature_selection import SelectKBest, chi2, f_regression 
  data = pd.read_csv('covid.train.csv')
  x = data[data.columns[2:117]]
  y = data[data.columns[117]]
  y = y.astype('int')
  
  bf1 = SelectKBest(score_func = chi2, k = 5)
  bf2 = SelectKBest(score_func = f_regression, k = 5)
  fit1 = bf1.fit(x, y)
  fit2 = bf2.fit(x, y)
  dfscores1 = pd.DataFrame(fit1.scores_)
  dfscores2 = pd.DataFrame(fit2.scores_)
  dfcolumns = pd.DataFrame(x.columns)
  
  #concat two dataframes for better visualization 
  featureScores1 = pd.concat([dfcolumns,dfscores1],axis=1)
  featureScores1.columns = ['Specs','Score']  #naming the dataframe columns
  print(featureScores1.nlargest(15,'Score'))  #print 15 best features
  ```

  挑选出feature

  ```python
                   Specs         Score
  103   nohh_cmnty_cli.4  11583.575780
  87    nohh_cmnty_cli.3  11426.105799
  71    nohh_cmnty_cli.2  11245.193098
  55    nohh_cmnty_cli.1  11043.387378
  99   tested_positive.3  11020.788317
  39      nohh_cmnty_cli  10817.588387
  83   tested_positive.2  10638.075755
  67   tested_positive.1  10231.222573
  102     hh_cmnty_cli.4  10210.937138
  86      hh_cmnty_cli.3  10041.724405
  70      hh_cmnty_cli.2   9853.471159
  51     tested_positive   9834.243513
  54      hh_cmnty_cli.1   9654.905496
  38        hh_cmnty_cli   9433.746008
  40        wearing_mask    721.849441
  111   public_transit.4    675.392064
  95    public_transit.3    665.973389
  79    public_transit.2    656.724054
  63    public_transit.1    650.615832
  47      public_transit    643.110897
  ```

  选了前14个。

- optimizer的选择

  - Adam
  - 

- L2 regularization

  

### HW2

序列标注问题应该说是NLP中最常见的问题。在深度学习没有广泛渗透到各个应用领域之前，传统的最常用的解决序列标注问题的方案是最大熵、CRF等模型，尤其是CRF，基本是最主流的方法。

所谓“序列标注”，就是说对于一个一维线性输入序列：

给线性序列中的每个元素打上标签集合中的某个标签