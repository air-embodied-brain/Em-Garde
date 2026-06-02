(video_understanding) aiot@aiot:~/mingjuwang/Em-Garde$ python -m arm_demo -
-yaml-path=configs/demo/demo_solution.yaml
[2026-05-21 20:01:15,265] DEBUG [git.cmd:1253] Popen(['git', 'version'], cwd=/home/aiot/mingjuwang/Em-Garde, stdin=None, shell=False, universal_newlines=False)
[2026-05-21 20:01:15,268] DEBUG [git.cmd:1253] Popen(['git', 'version'], cwd=/home/aiot/mingjuwang/Em-Garde, stdin=None, shell=False, universal_newlines=False)
[2026-05-21 20:01:15,314] DEBUG [wandb.docker.auth:50] Trying paths: ['/home/aiot/.docker/config.json', '/home/aiot/.dockercfg']
[2026-05-21 20:01:15,315] DEBUG [wandb.docker.auth:57] No config file found
/home/aiot/miniconda3/envs/video_understanding/lib/python3.10/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
DropoutAddRMSNorm of flash_attn is not installed!!!
/home/aiot/mingjuwang/Em-Garde/vlm2vec/model/baseline_backbone/internvideo2/modeling_internvideo2.py:539: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @torch.cuda.amp.autocast(enabled=False)
[2026-05-21 20:01:15,619] INFO [vlm2vec.model.vlm_backbone.qwen2_vl.qwen_vl_utils:41] set VIDEO_TOTAL_PIXELS: 90316800
[2026-05-21 20:01:15,672] DEBUG [matplotlib:337] matplotlib data path: /home/aiot/miniconda3/envs/video_understanding/lib/python3.10/site-packages/matplotlib/mpl-data
[2026-05-21 20:01:15,676] DEBUG [matplotlib:337] CONFIGDIR=/home/aiot/.config/matplotlib
[2026-05-21 20:01:15,677] DEBUG [matplotlib:1498] interactive is False
[2026-05-21 20:01:15,677] DEBUG [matplotlib:1499] platform is linux
[2026-05-21 20:01:15,715] DEBUG [matplotlib:337] CACHEDIR=/home/aiot/.cache/matplotlib
[2026-05-21 20:01:15,717] DEBUG [matplotlib.font_manager:1574] Using fontManager instance from /home/aiot/.cache/matplotlib/fontlist-v330.json
============================================================
Em-Garde ARM Demo
============================================================
Using FFmpeg for video reading (replaces decord)
Using OpenCV for video writing (replaces torchvision.io.write_video)
============================================================
Using device: cuda
You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
You have video processor config saved in `preprocessor.json` file which is deprecated. Video processor configs should be saved in their own `video_preprocessor.json` file. You can rename the file or load and save the processor back which renames it automatically. Loading from `preprocessor.json` will be removed in v5.0.
Loading checkpoint shards: 100%|█████████████| 4/4 [00:00<00:00,  8.83it/s]
Loading checkpoint shards: 100%|█████████████| 4/4 [00:00<00:00, 23.70it/s]
Unused or unrecognized kwargs: return_tensors, fps.
The following generation flags are not valid and may be ignored: ['temperature']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
At time 10.00s, processed query: What are the steps to make the solution?, given proposals: [{'positive': 'A hand uses a clear pipette to drip liquid into a clear container'}, {'positive': 'The clear container is then shaken or stirred'}, {'positive': 'The final clear container is shown with the mixed solution'}, {'positive': 'The hands hold a clear pipette and a clear container while adding the solution'}, {'positive': 'The clear container is poured into another clear container'}]
Processing timestep:10.00s, Time taken for processing: 15.6526s
Unused or unrecognized kwargs: return_tensors, fps.
The following generation flags are not valid and may be ignored: ['temperature']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
updating query proposals...
At time 10.00s, updated proposals for query 'What are the steps to make the solution?': [{'positive': 'A hand uses a clear pipette to drip liquid into a clear container'}, {'positive': 'The clear container is then shaken or stirred'}, {'positive': 'The final clear container is shown with the mixed solution'}, {'positive': 'The hands hold a clear pipette and a clear container while adding the solution'}, {'positive': 'The clear container is poured into another clear container'}]
Processing timestep:10.20s, Time taken for processing: 0.1646s
Processing timestep:10.40s, Time taken for processing: 0.1452s
Processing timestep:10.60s, Time taken for processing: 0.1467s
Processing timestep:10.80s, Time taken for processing: 0.1467s
Processing timestep:11.00s, Time taken for processing: 0.1459s
Processing timestep:11.20s, Time taken for processing: 0.1479s
Processing timestep:11.40s, Time taken for processing: 0.1458s
Unused or unrecognized kwargs: return_tensors, fps.
answer: Pour the clear liquid into the clear container.  check output: YES
Response generated for query: What are the steps to make the solution? at time 11.599999999999994s: Pour the clear liquid into the clear container.
Processing timestep:11.60s, Time taken for processing: 1.8466s
Processing timestep:11.80s, Time taken for processing: 0.1515s
Processing timestep:12.00s, Time taken for processing: 0.1492s
Processing timestep:12.20s, Time taken for processing: 0.1496s
Processing timestep:12.40s, Time taken for processing: 0.1519s
Processing timestep:12.60s, Time taken for processing: 0.1502s
Processing timestep:12.80s, Time taken for processing: 0.1473s
Processing timestep:13.00s, Time taken for processing: 0.1496s
Processing timestep:13.20s, Time taken for processing: 0.1497s
Processing timestep:13.40s, Time taken for processing: 0.1512s
Processing timestep:13.60s, Time taken for processing: 0.1511s
Processing timestep:13.80s, Time taken for processing: 0.1498s
Processing timestep:14.00s, Time taken for processing: 0.1515s
Processing timestep:14.20s, Time taken for processing: 0.1400s
Processing timestep:14.40s, Time taken for processing: 0.1411s
Processing timestep:14.60s, Time taken for processing: 0.1429s
Processing timestep:14.80s, Time taken for processing: 0.1463s
Processing timestep:15.00s, Time taken for processing: 0.1458s
Processing timestep:15.20s, Time taken for processing: 0.1454s
Processing timestep:15.40s, Time taken for processing: 0.1422s
Processing timestep:15.60s, Time taken for processing: 0.1421s
Processing timestep:15.80s, Time taken for processing: 0.1433s
Processing timestep:16.00s, Time taken for processing: 0.1403s
Processing timestep:16.20s, Time taken for processing: 0.1420s
Processing timestep:16.40s, Time taken for processing: 0.1417s
Processing timestep:16.60s, Time taken for processing: 0.1411s
Processing timestep:16.80s, Time taken for processing: 0.1412s
Processing timestep:17.00s, Time taken for processing: 0.1431s
Processing timestep:17.20s, Time taken for processing: 0.1425s
Processing timestep:17.40s, Time taken for processing: 0.1394s
Processing timestep:17.60s, Time taken for processing: 0.1430s
Processing timestep:17.80s, Time taken for processing: 0.1368s
Processing timestep:18.00s, Time taken for processing: 0.1432s
Processing timestep:18.20s, Time taken for processing: 0.1439s
Processing timestep:18.40s, Time taken for processing: 0.1450s
Processing timestep:18.60s, Time taken for processing: 0.1426s
Processing timestep:18.80s, Time taken for processing: 0.1432s
Processing timestep:19.00s, Time taken for processing: 0.1426s
Processing timestep:19.20s, Time taken for processing: 0.1431s
Processing timestep:19.40s, Time taken for processing: 0.1441s
Unused or unrecognized kwargs: return_tensors, fps.
answer: Pour the clear liquid into the clear container.  check output: NO
Processing timestep:19.60s, Time taken for processing: 1.7557s
Processing timestep:19.80s, Time taken for processing: 0.1393s
Processing timestep:20.00s, Time taken for processing: 0.1401s
Processing timestep:20.20s, Time taken for processing: 0.1435s
Processing timestep:20.40s, Time taken for processing: 0.1416s
Processing timestep:20.60s, Time taken for processing: 0.1436s
Processing timestep:20.80s, Time taken for processing: 0.1376s
Processing timestep:21.00s, Time taken for processing: 0.1418s
Processing timestep:21.20s, Time taken for processing: 0.1410s
Processing timestep:21.40s, Time taken for processing: 0.1399s
Processing timestep:21.60s, Time taken for processing: 0.1406s
Processing timestep:21.80s, Time taken for processing: 0.1431s
Processing timestep:22.00s, Time taken for processing: 0.1432s
Processing timestep:22.20s, Time taken for processing: 0.1421s
Processing timestep:22.40s, Time taken for processing: 0.1419s
Processing timestep:22.60s, Time taken for processing: 0.1447s
Processing timestep:22.80s, Time taken for processing: 0.1417s
Processing timestep:23.00s, Time taken for processing: 0.1414s
Processing timestep:23.20s, Time taken for processing: 0.1430s
Processing timestep:23.40s, Time taken for processing: 0.1421s
Processing timestep:23.60s, Time taken for processing: 0.1435s
Processing timestep:23.80s, Time taken for processing: 0.1424s
Processing timestep:24.00s, Time taken for processing: 0.1433s
Processing timestep:24.20s, Time taken for processing: 0.1431s
Processing timestep:24.40s, Time taken for processing: 0.1410s
Processing timestep:24.60s, Time taken for processing: 0.1421s
Processing timestep:24.80s, Time taken for processing: 0.1412s
Processing timestep:25.00s, Time taken for processing: 0.1447s
Processing timestep:25.20s, Time taken for processing: 0.1422s
Processing timestep:25.40s, Time taken for processing: 0.1414s
Processing timestep:25.60s, Time taken for processing: 0.1438s
Processing timestep:25.80s, Time taken for processing: 0.1442s
Processing timestep:26.00s, Time taken for processing: 0.1317s
Processing timestep:26.20s, Time taken for processing: 0.1419s
Processing timestep:26.40s, Time taken for processing: 0.1418s
Processing timestep:26.60s, Time taken for processing: 0.1444s
Processing timestep:26.80s, Time taken for processing: 0.1440s
Processing timestep:27.00s, Time taken for processing: 0.1435s
Processing timestep:27.20s, Time taken for processing: 0.1382s
Processing timestep:27.40s, Time taken for processing: 0.1430s
Processing timestep:27.60s, Time taken for processing: 0.1399s
Processing timestep:27.80s, Time taken for processing: 0.1406s
Processing timestep:28.00s, Time taken for processing: 0.1415s
Processing timestep:28.20s, Time taken for processing: 0.1420s
Processing timestep:28.40s, Time taken for processing: 0.1439s
Processing timestep:28.60s, Time taken for processing: 0.1435s
Processing timestep:28.80s, Time taken for processing: 0.1457s
Processing timestep:29.00s, Time taken for processing: 0.1365s
Processing timestep:29.20s, Time taken for processing: 0.1406s
Processing timestep:29.40s, Time taken for processing: 0.1425s
Processing timestep:29.60s, Time taken for processing: 0.1417s
Processing timestep:29.80s, Time taken for processing: 0.1433s
Processing timestep:30.00s, Time taken for processing: 0.1442s
Processing timestep:30.20s, Time taken for processing: 0.1448s
Processing timestep:30.40s, Time taken for processing: 0.1409s
Processing timestep:30.60s, Time taken for processing: 0.1434s
Processing timestep:30.80s, Time taken for processing: 0.1441s
Processing timestep:31.00s, Time taken for processing: 0.1427s
Processing timestep:31.20s, Time taken for processing: 0.1439s
Processing timestep:31.40s, Time taken for processing: 0.1445s
Processing timestep:31.60s, Time taken for processing: 0.1445s
Processing timestep:31.80s, Time taken for processing: 0.1418s
Processing timestep:32.00s, Time taken for processing: 0.1417s
Processing timestep:32.20s, Time taken for processing: 0.1431s
Processing timestep:32.40s, Time taken for processing: 0.1433s
Processing timestep:32.60s, Time taken for processing: 0.1368s
Processing timestep:32.80s, Time taken for processing: 0.1427s
Processing timestep:33.00s, Time taken for processing: 0.1420s
Processing timestep:33.20s, Time taken for processing: 0.1422s
Unused or unrecognized kwargs: return_tensors, fps.
answer: A hand uses a clear pipette to add a clear liquid into a clear container.  check output: YES
Response generated for query: What are the steps to make the solution? at time 33.39999999999994s: A hand uses a clear pipette to add a clear liquid into a clear container.
Processing timestep:33.40s, Time taken for processing: 2.7601s
Processing timestep:33.60s, Time taken for processing: 0.1491s
Processing timestep:33.80s, Time taken for processing: 0.1431s
Processing timestep:34.00s, Time taken for processing: 0.1425s
Processing timestep:34.20s, Time taken for processing: 0.1439s
Processing timestep:34.40s, Time taken for processing: 0.1442s
Processing timestep:34.60s, Time taken for processing: 0.1443s
Processing timestep:34.80s, Time taken for processing: 0.1410s
Processing timestep:35.00s, Time taken for processing: 0.1428s
Processing timestep:35.20s, Time taken for processing: 0.1408s
Processing timestep:35.40s, Time taken for processing: 0.1412s
Processing timestep:35.60s, Time taken for processing: 0.1440s
Processing timestep:35.80s, Time taken for processing: 0.1444s
Processing timestep:36.00s, Time taken for processing: 0.1461s
Processing timestep:36.20s, Time taken for processing: 0.1445s
Processing timestep:36.40s, Time taken for processing: 0.1430s
Processing timestep:36.60s, Time taken for processing: 0.1435s
Processing timestep:36.80s, Time taken for processing: 0.1412s
Processing timestep:37.00s, Time taken for processing: 0.1431s
Processing timestep:37.20s, Time taken for processing: 0.1433s
Processing timestep:37.40s, Time taken for processing: 0.1421s
Processing timestep:37.60s, Time taken for processing: 0.1438s
Processing timestep:37.80s, Time taken for processing: 0.1435s
Processing timestep:38.00s, Time taken for processing: 0.1441s
Processing timestep:38.20s, Time taken for processing: 0.1443s
Processing timestep:38.40s, Time taken for processing: 0.1432s
Processing timestep:38.60s, Time taken for processing: 0.1423s
Processing timestep:38.80s, Time taken for processing: 0.1443s
Processing timestep:39.00s, Time taken for processing: 0.1430s
Unused or unrecognized kwargs: return_tensors, fps.
answer: Pour the clear liquid into the clear container while stirring with a clear plastic spoon.  check output: YES
Response generated for query: What are the steps to make the solution? at time 39.200000000000024s: Pour the clear liquid into the clear container while stirring with a clear plastic spoon.
Processing timestep:39.20s, Time taken for processing: 2.6684s
Processing timestep:39.40s, Time taken for processing: 0.1456s
Processing timestep:39.60s, Time taken for processing: 0.1446s
Processing timestep:39.80s, Time taken for processing: 0.1436s
Processing timestep:40.00s, Time taken for processing: 0.1419s
Unused or unrecognized kwargs: return_tensors, fps.
The following generation flags are not valid and may be ignored: ['temperature']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
updating query proposals...
At time 40.00s, updated proposals for query 'What are the steps to make the solution?': [{'positive': 'A hand uses a clear pipette to add a clear liquid into a clear container'}, {'positive': 'The clear container is then shaken or stirred'}, {'positive': 'The clear container is poured into another clear container'}, {'positive': 'The clear container is covered with a clear lid'}, {'positive': 'The clear container is placed under a faucet for rinsing'}]
Processing timestep:40.20s, Time taken for processing: 0.1493s
Processing timestep:40.40s, Time taken for processing: 0.1488s
Processing timestep:40.60s, Time taken for processing: 0.1496s
Processing timestep:40.80s, Time taken for processing: 0.1501s
Processing timestep:41.00s, Time taken for processing: 0.1489s
Processing timestep:41.20s, Time taken for processing: 0.1485s
Processing timestep:41.40s, Time taken for processing: 0.1510s
Processing timestep:41.60s, Time taken for processing: 0.1490s
Processing timestep:41.80s, Time taken for processing: 0.1505s
Processing timestep:42.00s, Time taken for processing: 0.1504s
Processing timestep:42.20s, Time taken for processing: 0.1491s
Processing timestep:42.40s, Time taken for processing: 0.1424s
Processing timestep:42.60s, Time taken for processing: 0.1402s
Processing timestep:42.80s, Time taken for processing: 0.1415s
Processing timestep:43.00s, Time taken for processing: 0.1398s
Processing timestep:43.20s, Time taken for processing: 0.1412s
Processing timestep:43.40s, Time taken for processing: 0.1412s
Processing timestep:43.60s, Time taken for processing: 0.1409s
Processing timestep:43.80s, Time taken for processing: 0.1423s
Processing timestep:44.00s, Time taken for processing: 0.1398s
Processing timestep:44.20s, Time taken for processing: 0.1415s
Unused or unrecognized kwargs: return_tensors, fps.
answer: Add the clear liquid into the clear container while stirring with the clear plastic spoon.  check output: YES
Response generated for query: What are the steps to make the solution? at time 44.4000000000001s: Add the clear liquid into the clear container while stirring with the clear plastic spoon.
Processing timestep:44.40s, Time taken for processing: 2.6771s
Processing timestep:44.60s, Time taken for processing: 0.1440s
Processing timestep:44.80s, Time taken for processing: 0.1444s
Processing timestep:45.00s, Time taken for processing: 0.1421s
Processing timestep:45.20s, Time taken for processing: 0.1415s
Processing timestep:45.40s, Time taken for processing: 0.1417s
Processing timestep:45.60s, Time taken for processing: 0.1431s
Processing timestep:45.80s, Time taken for processing: 0.1427s
Processing timestep:46.00s, Time taken for processing: 0.1403s
Processing timestep:46.20s, Time taken for processing: 0.1429s
Processing timestep:46.40s, Time taken for processing: 0.1424s
Processing timestep:46.60s, Time taken for processing: 0.1433s
Processing timestep:46.80s, Time taken for processing: 0.1411s
Processing timestep:47.00s, Time taken for processing: 0.1412s
Processing timestep:47.20s, Time taken for processing: 0.1457s
Processing timestep:47.40s, Time taken for processing: 0.1417s
Processing timestep:47.60s, Time taken for processing: 0.1402s
Processing timestep:47.80s, Time taken for processing: 0.1425s
Processing timestep:48.00s, Time taken for processing: 0.1447s
Processing timestep:48.20s, Time taken for processing: 0.1408s
Processing timestep:48.40s, Time taken for processing: 0.1417s
Processing timestep:48.60s, Time taken for processing: 0.1450s
Processing timestep:48.80s, Time taken for processing: 0.1426s
Processing timestep:49.00s, Time taken for processing: 0.1419s
Processing timestep:49.20s, Time taken for processing: 0.1419s
Processing timestep:49.40s, Time taken for processing: 0.1417s
Processing timestep:49.60s, Time taken for processing: 0.1425s
Processing timestep:49.80s, Time taken for processing: 0.1421s



(video_understanding) aiot@aiot:~/mingjuwang/Em-Garde$ python -m arm_demo --yaml-path=configs/demo/demo_cat.yaml
[2026-05-28 19:25:17,530] DEBUG [git.cmd:1253] Popen(['git', 'version'], cwd=/home/aiot/mingjuwang/Em-Garde, stdin=None, shell=False, universal_newlines=False)
[2026-05-28 19:25:17,532] DEBUG [git.cmd:1253] Popen(['git', 'version'], cwd=/home/aiot/mingjuwang/Em-Garde, stdin=None, shell=False, universal_newlines=False)
[2026-05-28 19:25:17,578] DEBUG [wandb.docker.auth:50] Trying paths: ['/home/aiot/.docker/config.json', '/home/aiot/.dockercfg']
[2026-05-28 19:25:17,578] DEBUG [wandb.docker.auth:57] No config file found
/home/aiot/miniconda3/envs/video_understanding/lib/python3.10/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
DropoutAddRMSNorm of flash_attn is not installed!!!
/home/aiot/mingjuwang/Em-Garde/vlm2vec/model/baseline_backbone/internvideo2/modeling_internvideo2.py:539: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @torch.cuda.amp.autocast(enabled=False)
[2026-05-28 19:25:17,874] INFO [vlm2vec.model.vlm_backbone.qwen2_vl.qwen_vl_utils:41] set VIDEO_TOTAL_PIXELS: 90316800
[2026-05-28 19:25:17,927] DEBUG [matplotlib:337] matplotlib data path: /home/aiot/miniconda3/envs/video_understanding/lib/python3.10/site-packages/matplotlib/mpl-data
[2026-05-28 19:25:17,930] DEBUG [matplotlib:337] CONFIGDIR=/home/aiot/.config/matplotlib
[2026-05-28 19:25:17,931] DEBUG [matplotlib:1498] interactive is False
[2026-05-28 19:25:17,931] DEBUG [matplotlib:1499] platform is linux
[2026-05-28 19:25:17,969] DEBUG [matplotlib:337] CACHEDIR=/home/aiot/.cache/matplotlib
[2026-05-28 19:25:17,971] DEBUG [matplotlib.font_manager:1574] Using fontManager instance from /home/aiot/.cache/matplotlib/fontlist-v330.json
============================================================
Em-Garde ARM Demo
============================================================
Using FFmpeg for video reading (replaces decord)
Using OpenCV for video writing (replaces torchvision.io.write_video)
============================================================
Using device: cuda
You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
You have video processor config saved in `preprocessor.json` file which is deprecated. Video processor configs should be saved in their own `video_preprocessor.json` file. You can rename the file or load and save the processor back which renames it automatically. Loading from `preprocessor.json` will be removed in v5.0.
Loading checkpoint shards: 100%|██████████████████| 4/4 [00:00<00:00,  8.32it/s]
Loading checkpoint shards: 100%|██████████████████| 4/4 [00:00<00:00, 25.13it/s]
Unused or unrecognized kwargs: fps, return_tensors.
The following generation flags are not valid and may be ignored: ['temperature']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
At time 5.00s, processed query: Tell me when my steak is eaten by my cat!, given proposals: [{'positive': 'A hand uses a fork or spatula to lift a piece of steak from a pan and feed it to the cat'}, {'positive': 'The cat takes a bite of the steak while sitting at the table'}, {'positive': 'The person puts a piece of steak into a bowl for the cat to eat'}, {'positive': 'The cat chews on a bone while sitting near the table'}, {'negative': 'The person uses a spoon to scoop food into a pan'}]
Processing timestep:5.00s, Time taken for processing: 17.6742s
Unused or unrecognized kwargs: fps, return_tensors.
The following generation flags are not valid and may be ignored: ['temperature']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
updating query proposals...
At time 5.00s, updated proposals for query 'Tell me when my steak is eaten by my cat!': [{'positive': 'A hand uses a fork or spatula to lift a piece of steak from a pan and feed it to the cat'}, {'positive': 'The cat takes a bite of the steak while sitting at the table'}, {'positive': 'The person puts a piece of steak into a bowl for the cat to eat'}, {'positive': 'The cat chews on a bone while sitting near the table'}, {'negative': 'The person uses a spoon to scoop food into a pan'}]
Processing timestep:5.20s, Time taken for processing: 0.1273s
Processing timestep:5.40s, Time taken for processing: 0.1072s
Processing timestep:5.60s, Time taken for processing: 0.1081s
Processing timestep:5.80s, Time taken for processing: 0.1058s
Processing timestep:6.00s, Time taken for processing: 0.1050s
Processing timestep:6.20s, Time taken for processing: 0.1063s
Processing timestep:6.40s, Time taken for processing: 0.1076s
Processing timestep:6.60s, Time taken for processing: 0.1060s
Processing timestep:6.80s, Time taken for processing: 0.1055s
Processing timestep:7.00s, Time taken for processing: 0.1059s
Processing timestep:7.20s, Time taken for processing: 0.1054s
Processing timestep:7.40s, Time taken for processing: 0.1068s
Processing timestep:7.60s, Time taken for processing: 0.1073s
Processing timestep:7.80s, Time taken for processing: 0.1061s
Processing timestep:8.00s, Time taken for processing: 0.1056s
Processing timestep:8.20s, Time taken for processing: 0.1053s
Processing timestep:8.40s, Time taken for processing: 0.1063s
Unused or unrecognized kwargs: fps, return_tensors.
answer: The cat takes a bite of the steak from the pan with its mouth.  check output: YES
Response generated for query: Tell me when my steak is eaten by my cat! at time 8.6s: The cat takes a bite of the steak from the pan with its mouth.
Processing timestep:8.60s, Time taken for processing: 2.5997s
Processing timestep:8.80s, Time taken for processing: 0.1126s
Processing timestep:9.00s, Time taken for processing: 0.1096s
Processing timestep:9.20s, Time taken for processing: 0.1102s
Processing timestep:9.40s, Time taken for processing: 0.1092s
Processing timestep:9.60s, Time taken for processing: 0.1088s
Processing timestep:9.80s, Time taken for processing: 0.1063s
Processing timestep:10.00s, Time taken for processing: 0.1083s
Processing timestep:10.20s, Time taken for processing: 0.1081s
Processing timestep:10.40s, Time taken for processing: 0.1066s
Processing timestep:10.60s, Time taken for processing: 0.1061s
Processing timestep:10.80s, Time taken for processing: 0.1067s
Processing timestep:11.00s, Time taken for processing: 0.1055s
Processing timestep:11.20s, Time taken for processing: 0.1067s
Processing timestep:11.40s, Time taken for processing: 0.1043s
Processing timestep:11.60s, Time taken for processing: 0.1046s
Processing timestep:11.80s, Time taken for processing: 0.1078s
Processing timestep:12.00s, Time taken for processing: 0.1061s
Processing timestep:12.20s, Time taken for processing: 0.1038s
Processing timestep:12.40s, Time taken for processing: 0.1060s
Processing timestep:12.60s, Time taken for processing: 0.1084s
Processing timestep:12.80s, Time taken for processing: 0.1072s
Processing timestep:13.00s, Time taken for processing: 0.1073s
Processing timestep:13.20s, Time taken for processing: 0.1075s
Processing timestep:13.40s, Time taken for processing: 0.1059s
Processing timestep:13.60s, Time taken for processing: 0.1059s
Processing timestep:13.80s, Time taken for processing: 0.1083s
Processing timestep:14.00s, Time taken for processing: 0.1088s
Processing timestep:14.20s, Time taken for processing: 0.1061s
Processing timestep:14.40s, Time taken for processing: 0.1077s
Processing timestep:14.60s, Time taken for processing: 0.1063s
Processing timestep:14.80s, Time taken for processing: 0.1049s
Processing timestep:15.00s, Time taken for processing: 0.1045s
Processing timestep:15.20s, Time taken for processing: 0.1060s
Processing timestep:15.40s, Time taken for processing: 0.1037s
Processing timestep:15.60s, Time taken for processing: 0.1068s
Processing timestep:15.80s, Time taken for processing: 0.1075s
Processing timestep:16.00s, Time taken for processing: 0.1053s
Processing timestep:16.20s, Time taken for processing: 0.1055s
Processing timestep:16.40s, Time taken for processing: 0.1071s
Processing timestep:16.60s, Time taken for processing: 0.1065s
Processing timestep:16.80s, Time taken for processing: 0.1064s
Processing timestep:17.00s, Time taken for processing: 0.1084s
Processing timestep:17.20s, Time taken for processing: 0.1051s
Processing timestep:17.40s, Time taken for processing: 0.1054s
Processing timestep:17.60s, Time taken for processing: 0.1055s
Processing timestep:17.80s, Time taken for processing: 0.1066s
Processing timestep:18.00s, Time taken for processing: 0.1068s
Processing timestep:18.20s, Time taken for processing: 0.1069s
Processing timestep:18.40s, Time taken for processing: 0.1066s
Processing timestep:18.60s, Time taken for processing: 0.1057s
Processing timestep:18.80s, Time taken for processing: 0.1043s
Processing timestep:19.00s, Time taken for processing: 0.1056s
Processing timestep:19.20s, Time taken for processing: 0.1042s
Processing timestep:19.40s, Time taken for processing: 0.1036s
Processing timestep:19.60s, Time taken for processing: 0.1036s
Processing timestep:19.80s, Time taken for processing: 0.1043s
Processing timestep:20.00s, Time taken for processing: 0.1045s
Processing timestep:20.20s, Time taken for processing: 0.1027s
Processing timestep:20.40s, Time taken for processing: 0.1066s
Processing timestep:20.60s, Time taken for processing: 0.1051s
Processing timestep:20.80s, Time taken for processing: 0.1044s
Processing timestep:21.00s, Time taken for processing: 0.1049s
Processing timestep:21.20s, Time taken for processing: 0.1046s
Processing timestep:21.40s, Time taken for processing: 0.1066s
Processing timestep:21.60s, Time taken for processing: 0.1049s
Processing timestep:21.80s, Time taken for processing: 0.1066s
Unused or unrecognized kwargs: fps, return_tensors.
answer: The cat uses its paws to lift a piece of steak from a pan and then eats it.  check output: YES
Response generated for query: Tell me when my steak is eaten by my cat! at time 21.999999999999954s: The cat uses its paws to lift a piece of steak from a pan and then eats it.
Processing timestep:22.00s, Time taken for processing: 3.1277s
Processing timestep:22.20s, Time taken for processing: 0.1167s
Processing timestep:22.40s, Time taken for processing: 0.1097s
Processing timestep:22.60s, Time taken for processing: 0.1086s
Processing timestep:22.80s, Time taken for processing: 0.1060s
Processing timestep:23.00s, Time taken for processing: 0.1056s
Processing timestep:23.20s, Time taken for processing: 0.1038s
Processing timestep:23.40s, Time taken for processing: 0.1045s
Processing timestep:23.60s, Time taken for processing: 0.1040s
Processing timestep:23.80s, Time taken for processing: 0.1034s
Processing timestep:24.00s, Time taken for processing: 0.1044s
Processing timestep:24.20s, Time taken for processing: 0.1054s
Processing timestep:24.40s, Time taken for processing: 0.1063s
Processing timestep:24.60s, Time taken for processing: 0.1055s
Processing timestep:24.80s, Time taken for processing: 0.1058s
Processing timestep:25.00s, Time taken for processing: 0.1045s
Processing timestep:25.20s, Time taken for processing: 0.1047s
Processing timestep:25.40s, Time taken for processing: 0.1071s
Processing timestep:25.60s, Time taken for processing: 0.1088s
Processing timestep:25.80s, Time taken for processing: 0.1057s
Processing timestep:26.00s, Time taken for processing: 0.1061s
Processing timestep:26.20s, Time taken for processing: 0.1057s
Processing timestep:26.40s, Time taken for processing: 0.1047s
Processing timestep:26.60s, Time taken for processing: 0.1056s
Processing timestep:26.80s, Time taken for processing: 0.1039s
Processing timestep:27.00s, Time taken for processing: 0.1053s
Processing timestep:27.20s, Time taken for processing: 0.1049s
Processing timestep:27.40s, Time taken for processing: 0.1099s
Processing timestep:27.60s, Time taken for processing: 0.1060s
Processing timestep:27.80s, Time taken for processing: 0.1072s
Processing timestep:28.00s, Time taken for processing: 0.1078s
Processing timestep:28.20s, Time taken for processing: 0.1070s
Processing timestep:28.40s, Time taken for processing: 0.1049s
Processing timestep:28.60s, Time taken for processing: 0.1069s
Processing timestep:28.80s, Time taken for processing: 0.1078s
Processing timestep:29.00s, Time taken for processing: 0.1057s
Processing timestep:29.20s, Time taken for processing: 0.1067s
Processing timestep:29.40s, Time taken for processing: 0.1072s
Processing timestep:29.60s, Time taken for processing: 0.1070s
Processing timestep:29.80s, Time taken for processing: 0.1061s
Processing timestep:30.00s, Time taken for processing: 0.1055s
Processing timestep:30.20s, Time taken for processing: 0.1086s
Processing timestep:30.40s, Time taken for processing: 0.1056s
Processing timestep:30.60s, Time taken for processing: 0.1063s
Processing timestep:30.80s, Time taken for processing: 0.1069s
Processing timestep:31.00s, Time taken for processing: 0.1049s
Processing timestep:31.20s, Time taken for processing: 0.1063s
Processing timestep:31.40s, Time taken for processing: 0.1044s
Processing timestep:31.60s, Time taken for processing: 0.1069s
Processing timestep:31.80s, Time taken for processing: 0.1081s
Processing timestep:32.00s, Time taken for processing: 0.1062s
Processing timestep:32.20s, Time taken for processing: 0.1093s
Processing timestep:32.40s, Time taken for processing: 0.1103s
Processing timestep:32.60s, Time taken for processing: 0.1071s
Processing timestep:32.80s, Time taken for processing: 0.1053s
Processing timestep:33.00s, Time taken for processing: 0.1043s
Processing timestep:33.20s, Time taken for processing: 0.1061s
Processing timestep:33.40s, Time taken for processing: 0.1054s
Processing timestep:33.60s, Time taken for processing: 0.1064s
Processing timestep:33.80s, Time taken for processing: 0.1065s
Processing timestep:34.00s, Time taken for processing: 0.1049s
Processing timestep:34.20s, Time taken for processing: 0.1067s
Processing timestep:34.40s, Time taken for processing: 0.1049s
Processing timestep:34.60s, Time taken for processing: 0.1067s
Processing timestep:34.80s, Time taken for processing: 0.1070s
Processing timestep:35.00s, Time taken for processing: 0.1065s
Unused or unrecognized kwargs: fps, return_tensors.
The following generation flags are not valid and may be ignored: ['temperature']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
updating query proposals...
At time 35.00s, updated proposals for query 'Tell me when my steak is eaten by my cat!': [{'positive': 'A hand uses a spatula or fork to lift a piece of steak from a pan and feed it to the cat'}, {'positive': 'The cat takes a bite of the steak while sitting near the pan'}, {'positive': 'The pan of steak is shown without any hands or utensils'}, {'positive': 'The person puts a piece of steak into a bowl or plate'}, {'negative': 'The pan of steak is covered with a lid'}]
Processing timestep:35.20s, Time taken for processing: 0.1061s
Processing timestep:35.40s, Time taken for processing: 0.1069s
Processing timestep:35.60s, Time taken for processing: 0.1057s
Processing timestep:35.80s, Time taken for processing: 0.1050s
Processing timestep:36.00s, Time taken for processing: 0.1049s
Processing timestep:36.20s, Time taken for processing: 0.1057s
Processing timestep:36.40s, Time taken for processing: 0.1066s
Processing timestep:36.60s, Time taken for processing: 0.1064s
Processing timestep:36.80s, Time taken for processing: 0.1050s
Processing timestep:37.00s, Time taken for processing: 0.1049s
Processing timestep:37.20s, Time taken for processing: 0.1040s
Processing timestep:37.40s, Time taken for processing: 0.1070s
Processing timestep:37.60s, Time taken for processing: 0.1088s
Processing timestep:37.80s, Time taken for processing: 0.1076s
Processing timestep:38.00s, Time taken for processing: 0.1072s
Processing timestep:38.20s, Time taken for processing: 0.1061s
Processing timestep:38.40s, Time taken for processing: 0.1060s
Processing timestep:38.60s, Time taken for processing: 0.1055s
Unused or unrecognized kwargs: fps, return_tensors.
answer: The orange tabby cat uses its paws to lift a piece of steak off the black plate and then eats it.  check output: YES
Response generated for query: Tell me when my steak is eaten by my cat! at time 38.80000000000001s: The orange tabby cat uses its paws to lift a piece of steak off the black plate and then eats it.
Processing timestep:38.80s, Time taken for processing: 3.5975s
Processing timestep:39.00s, Time taken for processing: 0.1109s
Processing timestep:39.20s, Time taken for processing: 0.1077s
Processing timestep:39.40s, Time taken for processing: 0.1088s
Processing timestep:39.60s, Time taken for processing: 0.1040s
Processing timestep:39.80s, Time taken for processing: 0.1059s