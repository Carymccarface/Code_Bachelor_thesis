Requires download of the CGCNN from their gituhb:
   https://github.com/txie-93/cgcnn

Follow their readme instructions to initialize a CGCNN directory, the method used in the proejct was initalizing a new CGCNN directory for each target output.
Requires input POSCAR files

1. To use CGCNN first run either Setup_X.py file 
2. Then run poscar_conversion.py 
3. Make sure you are loading the correct POSCAR/Outcar Files and decide which property output you want according to your Setup_X.py

4. Then go into created cgcnn_X for either energy or magnetism
5. Then open data
6. Run preprocess.py 
7. Then run modify_json.py

8. Into the linux terminal copy:
"
 python3 main.py data   --task regression   --epochs 100   --batch-size 32   --train-ratio 0.75   --val-ratio 0.1   --test-ratio 0.15   --atom-fea-len 64   --h-fea-len 128   --n-conv 3   --n-h 1   --lr 0.001   --weight-decay 1e-5   --lr-milestones 50 80
"
Or other settings

9. Then run X_Graph.py to again get results and graphs 

10. To predict Energy for New Structures:
python predict.py --model-path model_best.pth.tar --data-path <folder_with_new_poscars>
