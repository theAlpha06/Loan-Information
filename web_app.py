from flask import Flask, render_template,request
import numpy as np
import os 
import joblib

app=Flask(__name__)

result=""

@app.route('/',methods=['GET'])
def welcome():
    return render_template("index1.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        X = []

        clsmodel = joblib.load('classification_model.joblib')

        rgrmodel = joblib.load('ridge_model.joblib')

        columns = ['CreditGrade','BorrowerAPR', 'BorrowerRate',
                   'LenderYield', 'ProsperScore', 'CreditScore', 'MonthlyLoanPayment',
                   'LP_CustomerPayments','LP_InterestandFees', 'LP_ServiceFees',
                   'LP_CustomerPrincipalPayments', 'LP_CollectionFees' , 
                   'LP_GrossPrincipalLoss', 'LoanOriginalAmount',
                   'StatedMonthlyIncome']

        for column in columns:
            value=request.form.get(column)
            # print(value)
            X.append(float(value))

        # print("X values = ", X)

        X = np.array(X)
        X = X.reshape(1, -1)
        test_arr = X

        yreg_pred = rgrmodel.predict(test_arr)
        ycls_pred = clsmodel.predict(test_arr)

        yreg_pred=np.array(yreg_pred)
        EMI = np.round(yreg_pred[:, 0],3)
        PROI = np.round(yreg_pred[:, 1],3)
        ELA = np.round(yreg_pred[:, 2],3)
        
        ELA=ELA/100

        # print("Cls Model Object: ", clsmodel)
        # print("Rgr Model Object: ", rgrmodel)


        if ycls_pred==0:
            predicted="Not Defaulted"
        else:
            predicted="Defaulted"
        result = f"""The model has predicted that the result for Loan Status is : {predicted}. \n 
                   The PROI = {str(PROI)[1:-1]}, ELA = {str(ELA)[1:-1]} and EMI = {str(EMI)[1:-1]}"""
                   
        return render_template('index1.html', result=result)
    return render_template('index1.html')

if __name__ == '__main__':
    app.run( debug=True)

