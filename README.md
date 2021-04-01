# sequence_analyzer
for sfmlearner dataset analysis

一共两个重要
```

|-main
  |- analyzer: 对序列进行计算, 包括帧间光度误差图, histogram等
  |- drawer: 对analyzer得到的计算结果来绘图

```
### analyzer

对图像序列进行计算, 例如根据50张连续帧, 得到帧间光度误差图, 输出为

```
class analyzer
  |- init
    #output: seq_any/log_name(dir)#wk_dir
      |- photometric_error_maps(dir)
      |- error_maps_hist(dir)
      

  |-photometric_error_map
	#产生多个序列的error maps 序列, 按照logs 输入
	#前置函数:None
	#setting files: settings.yaml
	#input: logs/log_name.txt
	#output: wk_dir/photometric_error_maps
	  |- 0000e_15k.npy
	  ...
	  |- 0000e_15k(dir)
	
  |- error_map_hist
    # analyzer.histogram_analysis
    #setting files: settings.yaml
    #前置函数: photometric_error_map
    #input: logs/log_name.txt
    #output: wk_dir/error_maps_hist
      |- 0000e_09k.npy
      ...
      

```

输入./logs/logs.txt