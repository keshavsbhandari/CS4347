**ASSIGNMENT 3 :** 



**IMPORTANT NOTICE : DEADLINE EXTENDED FOR ONE MORE WEEK**  
    
    GOOD NEWS : DUDE DATE IS NOW 26TH OF NOVEMBER
    
    I have realised you guys are having pretty bad time doing this assignment, as you are only relying on this
    pytorch project. But, to start easily keras is much more easier and you can implement it with few lines of codes.
    I have implemented two working  SIMPLE-CNNs for each case, hotdog-not-hotdog and food101small case. These two files 
    can be find inside assign3 folder as 
        - kerasmain.py [for hotdog-not-hotdog] and 
        - kerasmain101.py [for 10 class food101 dataset].
    
    before running this file make sure you have install following(in the same environment you have installed previous 
    dependencies, 
    WARNING : Windows user might have problem but thats your perk of using windows fix it by yourself or email me.) 
    
    pip install keras
    pip install tensorflow 
    
    you can run this by simply, 
        python kerasmain.py
        python kerasmain101.py
        
    remember after training for given epochs time these files will save the model to 
        models/kerasfoodsmall.h5 by kerasmain.py
        models/kerasfood101.h5 by kerasmain101.py
        
        [SO YOU CAN STUDY FROM KERAS DOCUMENTATION AND LOAD THIS MODEL IF YOU NEED IT]
        
        WARNING : YOU MIGHT CHANGE YOUR MODEL TO GAIN MORE ACCURACY AS A PART OF ASSIGNMENT, 
        IN THAT CASE PREVIOUSLY SAVED MODEL MIGHT NOT LOAD, SO BE CAREFUL TO SAVE OR LOAD MODEL
        BASED ON THE ARCHITECTURE YOU ARE USING
        
    moreover, our two different implementations does profiling as well
    you can see (accuracy, loss and model graph in tensorboard),
        tensorboard --logdir=foodsmallruns
        tensorboard --logdir=food101runs
    
    in kerasmain.py and kerasmain101.py you will see a class from keras ImageDataGenerator
    tryo to study this from the documentation and make new changes to it, there are many different options 
    you can use to enhance your training 
    
    Similarly another thing you can change is optimizer, now we are using RmsProp you can change this to 
    another optimizer and see how it behaves,
    similarly you can tweak different learning rate and other parameters in optimizer
    
    One important changes you can make is in network itself, you can add or remove convolution layers,
    pooling layers, dropout, batchnormalization and activations(with different kinds).
    
    I hope this will make your work more easier. 
         
    
    

**NEURAL NETWORK | CONVOLUTION | DEEP LEARNING**


    VISIT FOLLOWING KAGGLE SITE
    
    NOTE : IF YOU ARE USING GPU : FOLLOW RESPECTIVE FRAMEWORK(PYTORCH,TENSORFLOW OR KERAS) WEBSITE TO INSTALL GPU VERSION
    
    DOWNLOAD DATA SET FROM 
        https://www.kaggle.com/prathmeshgodse/food101-zip/kernels
    
    [OPTIONAL]
        STARTER CODE  TO UNDERSTAND DATA:
        https://www.kaggle.com/kerneler/starter-food101-zip-b3e63593-0
    
    Note
        [YOU ARE FREE TO USE YOUR OWN PROJECT STRUCTURE, HOWEVER THIS REPO CAN HELP YOU
        FOR STARTING]
        [YOU ARE FREE TO USE ANY FRAMEWORK PYTORCH,TENSORFLOW OR KERAS]
        
        [IF YOU ARE USING PYTORCH YOU CAN TAKE THIS REPO AS STARTER CODE]
    
        TASK 1: CLONE THIS REPO AND USE THIS PROJECT AS STARTER CODE[OPTIONAL]
        
        TASK 2: CONFIGURE[OPTIONAL] DATALOADER ACCORDING TO YOUR NEED OR USE YOUR OWN
        
        TASK 3: TRY TO TWEAK THE MODEL[OPTIONAL] TO FIT YOUR NEED OR DESIGN YOUR OWN MODEL OR USE PRETRAINED MODEL
                TIPS : RIGHT NOW MODEL IS IMPLEMENTED FOR 1 CLASS PROBLEM
        
        TASK 4: [OPTIONAL]IN TRAINER YOU NEED TO UPDATE YOUR LOSS FUNCTION
                TIPS : CHANGE IT TO FIT ON MULTICLASS CLASSIFICATION
                OR DESIGN YOUR OWN TRAINER
                
        TASK 5: [OPTIONAL]IN TRAINER YOU NEED TO UPDATE TRAINING LOOP,
                TIPS : YOU NEED TO CHANGE HOW ACCURACY IS CALCULATED
                OR DESIGN YOUR OWN TRAINER
        
        TASK 6: [OPTIONAL] IN TRAINER YOU NEED TO UPDATE TESTING LOOP, 
                TIPS : YOU NEED TO CHANGE HOW ACCURACY IS CALCULATED
                OR DESIGN YOUR OWN TRAINER
        
        TASK 7: RUN THE EXPERIMENT ON MINIMUM 10 CLASSES 
        
        TASK 8: CREATE A REPO OF YOUR PROJECT
        
        TASK 9: IN YOU REPORT SUBMIT FOLLOWING
                
                - LINK TO YOUR GITHUB REPO
                
                - BRIEF DETAILS OF ARCHITECTURE YOU USE
                        NOTE YOU CAN USE PRETRAINED ARCHITECTURE AS WELL
                
                - REPORTS OF EXPERIMENT RESULTS
                    ACCURACY 
                    GRAPHS OF LOSSES AND ACCURACY
                
                - REPORT FOLLOWING __ THAT YOU HAVE USED IN YOUR EXPERIMENT
                    - LOSS FUNCTION
                    - LEARNING RATE
                    - SCHEDULER (IF USED)
                    - OPTIMIZER
                    - EPOCH
                    - TRAIN SIZE
                    - TEST SIZE
                    - DATA DIMENSION HEIGHT X WIDTH
                    - NUMBER OF PARAMTERS OF YOUR MODEL [OPTIONAL]
                        IF YOU ARE USING THIS REPO AS YOUR STARTER CODE
                        you can do it manually by following code
                        in notebook
                        
                        from torchsummary import summary
                        from models.simpleclassifier import NaiveDLClassifier
                        
                        summary(NaiveDLClassifier(), input_size = (3,128,128), device = 'cpu')
                        
                        IF YOU ARE NOT USING THIS REPO OR USING KERAS, 
                        YOU CAN DO THIS USING KERAS API, SEARCH ON GOOGLE
                        
        NOTE : IF YOU ARE USING THIS REPO AS STARTER CODE YOU CAN CREATE AN ENVIRONMENT AND RUN FOLLOWING
                pip install -r requirements.txt
                
                It has all the dependencies you need
                
                NOTE : IF YOU ARE USING GPU : FOLLOW RESPECTIVE FRAMEWORK(PYTORCH,TENSORFLOW OR KERAS) WEBSITE 
                TO INSTALL GPU VERSION
                and if you are using pytorch with gpu, above pip install will install the cpu version, please remove pytorch                   and install gpu version again.
                      
                
                 
                
                         
        




