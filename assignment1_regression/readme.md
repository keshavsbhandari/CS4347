WELCOME TO ASSIGNMENT 1

TOTAL POINTS : 200

**IMPORTANT NOTICE**
    
    Please submit only report in a pdf format , anwering every part of the questions 
    Reports should include screenshots of visualization of plots you get when you run your project
    If you have any confusion please visit me(TA) during my office hour or write an email
    

`PART 1 : OBSERVATION/SETUP OF THE PROJECT [50 pts]` 

    1. Did you understand this project structure? What are the key things 
       you have noticed? [10 pts]
       
    2. Were you able to create environment using environment.yml 
       file present in the directory? [10 pts]
       
    3. You are encouraged to visit misc folder and play with datamanip.ipynb 
       this notebook has basic data science pipe line , that shwos you how should you 
       deal with real world data [10 pts]
       
       Please visit the ipynb file and study the nature of the data
       
       Did you complete exploring ipynb file? Why do you think it is important?
    
    4. In datamanip.ipynb datasets are standardize before saving to train-test split
       why it is necessary? [10 pts]
       
    5. Why do you think, we use following lambda operatoin [10 pts]
       E = lambda x: np.insert(x, 0, 1, axis=1)
       is used in following 
       ./models/SimpleRegression.py
       ./utils/RandomDataGenerator.py
       

`PART 2 : COMPLETE THE CODING [50 pts]` 

    You will be working on models/SimpleRegression.py file as your assignment
    
    NOTE : Write only block of code you need to complete as your answers in report
    
    HOW TO TEST YOUR CODE ??
        RUN IN TERMINAL : python main.py configs/ranuniconfig.txt
    
    NOTE YOU CAN WRITE MULTIPLE LINES OF CODE EXCEPT UNLESS MENTIONED EXPLICITLY
    
    submission report
    
    Question 1 : inside __init__ [10 pts]
        
    Quesiton 2 : inside gradient_descent [10 pts]
    
    Question 3 : inside __rmse [10 pts]
    
    Quesiton 4 : inside __r2_score [10 pts]
    
    Quesiton 5 : inside __get_cost [10 pts]

`PART 3 : AFTER YOU HAVE WRITTEN YOUR CODE PLAY WITH REAL DATA AND ANSWER FOLLOWING [70 pts]`
    
    NOTE : 
        realconfig.txt : configuration for realstate dataset
        vidconfig.txt : configuration for video encoding dataset
        ranconfig.txt : configuration for random dataset
        ranuniconfig.txt : configuration for only unimodal random dataset
        
        ???HOW TO USE CERTAIN CONFIG ???
            eg. using realconfig.txt
                run : python main.py realconfig.txt
        
        you can tweak the paramters in these confiurations
        
        Question 1 : Report results(only val and test) on ranuniconfig
                     tweak n, lambda and iterations and explain how these parameters 
                     effect your result.[10 pts]
                     
                     
        Question 2 : Report results(only val and test) on ranconfig.txt
                     tweak n,lambda,xdim and iterations and explain how these parameters 
                     effect your result.[10 pts]
                     
        
        Question 3 : Report results(only val and test) on realconfig.txt
                     tweak lmbda, alpha and iterations parameter and explain 
                     how these paramters effect your result.[10 pts]
        
        Question 4 : Report results(only val and test) on vidconfig.txt
                     tweak lmbda, alpha,batch_size and iterations paramter and explain 
                     how these parameters effect your result[10 pts]
                     
        
        Question 5 : Compare Question 1 and Question 2 results.[10 pts]
        
        Question 6 : Compare Question 3 and Question 4 results.[10 pts]
        
        Question 7 : If you have noticed from Question1 from this section, 
                     there is also an analytical solution used. You can find this 
                     solution in utils/RandomDataGenerator.py file (func : analytical)
                     
                     Why do we need machine learning approach though we can get solution 
                     from this approach, as you have noticed?
                     

`PART 4: THIS SECTION IS TO THANK YOU FOR YOUR PARTICIPATION [30 pts]` 
        
                     
                     
                     
     
       
        
       
