**ASSIGNMENT 3 :** 

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
                      
                
                 
                
                         
        




