==============================================================================
VAL_LOSS 改善历史（只显示有改善的 epoch）
==============================================================================
 epoch      step    val_loss        gain      累计 epoch
------------------------------------------------------------------------------
     1      7562     0.98457                         1
     2     11343     0.97060    +0.01397             2
     3     15124     0.95189    +0.01871             3
     4     18905     0.94212    +0.00977             4
     5     22686     0.92930    +0.01282             5
     6     26467     0.92169    +0.00761             6
     7     30248     0.91442    +0.00727             7
     8     34029     0.90700    +0.00742             8
    10     41591     0.90246    +0.00454            10
    12     49153     0.89396    +0.00850            12
    14     56715     0.89107    +0.00289            14
    16     64277     0.88839    +0.00268            16
    18     71839     0.88679    +0.00160            18
    19     75620     0.87721    +0.00958            19
    21     83182     0.87668    +0.00053            21
    23     90744     0.87401    +0.00267            23
    24     94525     0.87389    +0.00012            24
    25     98306     0.87097    +0.00292            25
    26    102087     0.86799    +0.00298            26
    29    113430     0.86334    +0.00465            29
    30    117211     0.86105    +0.00229            30
    33    128554     0.85697    +0.00408            33
    35    136116     0.85630    +0.00067            35
    37    143678     0.85620    +0.00010            37
    39    151240     0.85096    +0.00524            39
    42    162583     0.84936    +0.00160            42
    43    166364     0.84858    +0.00078            43
    45    173926     0.84618    +0.00240            45
    46    177707     0.84453    +0.00165            46
    47    181488     0.84320    +0.00133            47
    49    189050     0.83971    +0.00349            49
    52    200393     0.83703    +0.00268            52
    53    204174     0.83674    +0.00029            53
    54    207955     0.83253    +0.00421            54
    57    219298     0.82890    +0.00363            57
    63    241984     0.82203    +0.00687            63
    66    253327     0.82013    +0.00190            66
    68    260889     0.81716    +0.00297            68
    72    276013     0.81336    +0.00380            72
    75    287356     0.81292    +0.00044            75
    83    317604     0.80963    +0.00329            83
    84    321385     0.80783    +0.00180            84
    88    336509     0.80633    +0.00150            88
    91    347852     0.80459    +0.00174            91
    92    351633     0.80135    +0.00324            92
    98    374319     0.79260    +0.00875            98
   103    393224     0.79188    +0.00072           103
   115    438596     0.79085    +0.00103           115
   119    453720     0.78994    +0.00091           119
   120    457501     0.78787    +0.00207           120
   123    468844     0.78633    +0.00154           123
   125    476406     0.78495    +0.00138           125
   129    491530     0.78185    +0.00310           129
   132    502873     0.78105    +0.00080           132
   138    525559     0.78098    +0.00007           138
   143    544464     0.78012    +0.00086           143
   146    555807     0.77234    +0.00778           146
   159    604960     0.76945    +0.00289           159
   169    642770     0.76834    +0.00111           169
   175    665456     0.76827    +0.00007           175
   178    676799     0.76594    +0.00233           178
   179    680580     0.76591    +0.00003           179
   181    688142     0.76536    +0.00055           181
   183    695704     0.76461    +0.00075           183
   191    725952     0.76391    +0.00070           191
   192    729733     0.75923    +0.00468           192
   210    797791     0.75741    +0.00182           210
   213    809134     0.75468    +0.00273           213
   222    843163     0.75466    +0.00002           222
   229    869630     0.75002    +0.00464           229
   234    888535     0.74974    +0.00028           234
   248    941469     0.74490    +0.00484           248
   260    986841     0.74467    +0.00023           260
   273   1035994     0.73874    +0.00593           273
   298   1130519     0.73783    +0.00091           298
   307   1164548     0.73628    +0.00155           307
   328   1243949     0.73518    +0.00110           328
   352   1334693     0.73401    +0.00117           352
   366   1387627     0.72998    +0.00403           366

==============================================================================
最近改善趋势
==============================================================================
  最近 6 次改善：
    epoch 298: val=0.73783  improvement=+0.00091
    epoch 307: val=0.73628  improvement=+0.00155  █
    epoch 328: val=0.73518  improvement=+0.00110  █
    epoch 352: val=0.73401  improvement=+0.00117  █
    epoch 366: val=0.72998  improvement=+0.00403  ████

==============================================================================
早停状态
==============================================================================
  当前最佳 val_loss : 0.72998 (epoch 366)
  最新已完成 epoch  : 396
  自上次改善已过    : 30 个 epoch
  早停 patience     : 30
  距离早停还剩      : 0 个 epoch

  🛑 训练已经触发 EarlyStopping！

  日志最后更新于: 2637 秒前
  ⚠️  超过 5 分钟没更新，可能训练已停止
(jhub_env) tcat@scsmlnprd02:~$ echo "===== [1] 进程状态（应该已经退出）====="
ps -p $(cat /home/tcat/diffcsp_exp4/logs/step4_train.pid) -o pid,etime,cmd 2>&1

echo ""
echo "===== [2] GPU 是否释放 ====="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

echo ""
echo "===== [3] 最佳 ckpt 路径文件（脚本最后一步写的）====="
cat /home/tcat/diffcsp_exp4/best_checkpoint_path.txt 2>&1

echo ""
echo "===== [4] checkpoints 目录现状 ====="
ls -la /home/tcat/diffcsp_exp4/checkpoints/

echo ""
echo "===== [5] stderr 末尾 30 行（看训练完成信息）====="
tail -n 30 /home/tcat/diffcsp_exp4/logs/step4_train_stderr.log

echo ""
echo "===== [6] stdout 末尾 5 行（看 logger.info 输出）====="
tail -n 5 /home/tcat/diffcsp_exp4/logs/step4_train_stdout.log
===== [1] 进程状态（应该已经退出）=====
    PID     ELAPSED CMD

===== [2] GPU 是否释放 =====
pid, used_gpu_memory [MiB]

===== [3] 最佳 ckpt 路径文件（脚本最后一步写的）=====
/home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt
===== [4] checkpoints 目录现状 =====
total 78584
drwxrwxr-x 3 tcat tcat     4096 Apr 27 12:05 .
drwxrwxr-x 7 tcat tcat     4096 Apr 27 14:26 ..
-rw-rw-r-- 1 tcat tcat 40224914 Apr 27 12:05 best-epoch366-val0.7300.ckpt
-rw-rw-r-- 1 tcat tcat 40224914 Apr 27 14:26 last.ckpt
drwxrwxr-x 2 tcat tcat     4096 Apr 26 06:20 _smoke

===== [5] stderr 末尾 30 行（看训练完成信息）=====
2026-04-27 12:33:34,209  INFO  Epoch 372, global step 1410313: 'val_loss' was not in top 1
2026-04-27 12:38:14,491  INFO  Epoch 373, global step 1414094: 'val_loss' was not in top 1
2026-04-27 12:42:53,724  INFO  Epoch 374, global step 1417875: 'val_loss' was not in top 1
2026-04-27 12:47:34,164  INFO  Epoch 375, global step 1421656: 'val_loss' was not in top 1
2026-04-27 12:52:16,168  INFO  Epoch 376, global step 1425437: 'val_loss' was not in top 1
2026-04-27 12:56:54,844  INFO  Epoch 377, global step 1429218: 'val_loss' was not in top 1
2026-04-27 13:01:33,841  INFO  Epoch 378, global step 1432999: 'val_loss' was not in top 1
2026-04-27 13:06:13,265  INFO  Epoch 379, global step 1436780: 'val_loss' was not in top 1
2026-04-27 13:10:54,000  INFO  Epoch 380, global step 1440561: 'val_loss' was not in top 1
2026-04-27 13:15:35,532  INFO  Epoch 381, global step 1444342: 'val_loss' was not in top 1
2026-04-27 13:20:17,388  INFO  Epoch 382, global step 1448123: 'val_loss' was not in top 1
2026-04-27 13:25:00,139  INFO  Epoch 383, global step 1451904: 'val_loss' was not in top 1
2026-04-27 13:29:40,552  INFO  Epoch 384, global step 1455685: 'val_loss' was not in top 1
2026-04-27 13:34:24,229  INFO  Epoch 385, global step 1459466: 'val_loss' was not in top 1
2026-04-27 13:39:06,568  INFO  Epoch 386, global step 1463247: 'val_loss' was not in top 1
2026-04-27 13:43:51,875  INFO  Epoch 387, global step 1467028: 'val_loss' was not in top 1
2026-04-27 13:48:31,365  INFO  Epoch 388, global step 1470809: 'val_loss' was not in top 1
2026-04-27 13:53:14,255  INFO  Epoch 389, global step 1474590: 'val_loss' was not in top 1
2026-04-27 13:57:55,904  INFO  Epoch 390, global step 1478371: 'val_loss' was not in top 1
2026-04-27 14:02:40,876  INFO  Epoch 391, global step 1482152: 'val_loss' was not in top 1
2026-04-27 14:07:20,513  INFO  Epoch 392, global step 1485933: 'val_loss' was not in top 1
2026-04-27 14:12:01,240  INFO  Epoch 393, global step 1489714: 'val_loss' was not in top 1
2026-04-27 14:16:41,639  INFO  Epoch 394, global step 1493495: 'val_loss' was not in top 1
2026-04-27 14:21:24,698  INFO  Epoch 395, global step 1497276: 'val_loss' was not in top 1
2026-04-27 14:26:06,593  INFO  Monitored metric val_loss did not improve in the last 30 records. Best score: 0.730. Signaling Trainer to stop.
2026-04-27 14:26:06,594  INFO  Epoch 396, global step 1501057: 'val_loss' was not in top 1
2026-04-27 14:26:07,631  INFO  训练完成。
2026-04-27 14:26:07,631  INFO  最优 ckpt : /home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt
2026-04-27 14:26:07,631  INFO  最优 val_loss : 0.729984
2026-04-27 14:26:07,631  INFO  路径写入 → /home/tcat/diffcsp_exp4/best_checkpoint_path.txt

===== [6] stdout 末尾 5 行（看 logger.info 输出）=====
[XasLocalDatasetV2 benchmark] feff.loc avg: 53.60 µs/sample (N=1000); cutover threshold: 200 µs → consider dict cache (decision 7.5B)
[XasLocalDatasetV2 benchmark] POSCAR + SGA avg: 13.65 ms/sample (N=50); >50 ms → consider lru_cache after Step 4 profile
[XasLocalDatasetV2] cache LOADED for val: valid=7621/7624 from val_structure_cache.pt
[XasLocalDatasetV2] split=val samples=7624 ready.
[XasLocalDataModuleV2] train=60507 val=7624