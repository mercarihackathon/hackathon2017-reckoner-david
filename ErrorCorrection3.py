# -*- coding: utf-8 -*-
import os, sys
from sklearn import cross_validation

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from translate import Translate
from collections import defaultdict
from pandas import Series
import sys, os, re
import math
from datetime import datetime
import warnings
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from translate import Translate
import azure_translate_api
from langdetect import detect
from microsofttranslator import Translator
from textblob import TextBlob

class Correction:
    df_AllergenData = pd.DataFrame()
    df_AllergenCodes = pd.DataFrame()
    df_AllergenInformation = pd.DataFrame()
    IngredientsFrame = pd.DataFrame()
    df_preservativesInfo = pd.DataFrame()
    df_Translated = pd.DataFrame()
    
    df_GtinList = pd.DataFrame()
    
    
    df_ColumnHeaders = pd.DataFrame()

    def __init__(self):
        self.df_AllergenData = pd.read_csv('./DataFilesForDataProcessing/AllegenData.csv', encoding= 'latin-1')
        self.df_AllergenData = self.df_AllergenData.sort_values('Gtin')
        self.df_AllergenCodes = pd.read_csv('./DataFilesForDataProcessing/AleergenTypeCode.csv', encoding='latin-1')
        self.df_AllergenInformation = pd.read_csv('./DataFilesForDataProcessing/AllergenInformation.csv', encoding='latin-1')
        self.df_AllergenInformation = self.df_AllergenInformation.sort_values('Gtin')
        self.IngredientsFrame = pd.read_csv('./DataFilesForDataProcessing/AllegenData.csv', encoding= 'latin-1')
        self.IngredientsFrame = self.IngredientsFrame.sort_values('Gtin')
        self.df_preservativesInfo = pd.read_csv('./DataFilesForDataProcessing/AdditiveInformation.csv', encoding='latin-1')
        self.df_Translated = pd.read_csv('./DataFilesForDataProcessing/frame.csv', encoding='latin-1')
        self.IngredientsInString = self.IngredientsFrame['IngredientsInfo'].tolist()
        self.languages=self.IngredientsFrame['LanguageInfo'].tolist()
        self.temp=(self.IngredientsFrame['Gtin'].tolist())
        
        self.df_GtinList = pd.read_csv('./DataFilesForDataProcessing/GtinList.csv', encoding='latin-1')
        self.df_GtinList = self.df_GtinList.sort_values('ProductIDs')
        
        self.df_ColumnHeaders = pd.read_csv('./DataFilesForDataProcessing/ColumnHeaders.csv', encoding='latin-1')

    # def keywordframe(self):


    def SetUpFeatures(self):
        featuresSet = set()
        for i in range (len(self.IngredientsInString)):
            try:
               # translatedstring = translator(self.languages[i], 'en', self.IngredientsInString[i])
                #if (i%50==0): print(translatedstring)
                vals = re.findall(ur'\(?\bE\s*\d+\w*\)?\W*|\b([^\W\d]+(?:\s+[^\W\d]+)*)\b', self.IngredientsInString[i], re.UNICODE)
                self.ListOfIngredients = [x for x in vals if x]
                for j in range (len(self.ListOfIngredients)):
                    if(len(self.ListOfIngredients[j]) < 4): continue
                    featuresSet.add(self.ListOfIngredients[j].lower())
            except Exception as e:
                continue

        # CLIENT_ID = "fin2016myopt"
        # CLIENT_SECRET = "dK0cIVVGYmpAz1ynUpxp3h4ieBjjQUhNR0RLI8XLAn0="
        # translator = Translator('fin2016myopt', 'dK0cIVVGYmpAz1ynUpxp3h4ieBjjQUhNR0RLI8XLAn0=')


        ctr =0;
        featuresSet2 = set()
        for i in featuresSet:
            # print(i)
            # if (ctr%100 == 0): print (ctr/100)
            blob = TextBlob(i);
            ingredientwords = blob.words
            ingredientwords = ingredientwords.singularize()
            # print ingredientwords
            for j in ingredientwords:
                text = TextBlob(j)
                sentence  = text.tags
                # print sentence
                for a , postag in sentence:
                    if postag == 'NN': 
                        # print(a)
                        featuresSet2.add(a)
            ctr+=1


        # featuresSet2 = set()
        # ctr =0
        # for i in featuresSet:
        #     if (ctr == 0) : print (ctr)
        #     try :                
        #         c = translator.translate(i, 'en')
        #         print(i + '###' + c)
        #         featuresSet2.add(c)
        #     except Exception as e:
        #         print (e)
        #         featuresSet2.add(i)
        #     ctr += 1    

        # dbg=pd.DataFrame(list(featuresSet2))
        
        # dbg.to_csv('File6.csv', sep=',', encoding='latin-1', index = False)
        featuresSet2 = sorted(featuresSet2)
        return featuresSet2


    def SetUpDataFrameFromFeatures(self):
        gtinList = self.df_GtinList
        featureSet = list(self.SetUpFeatures())
        # productIds = self.IngredientsFrame['Gtin'].astype(str).tolist()
        productIds = gtinList['ProductIDs'].astype(str).tolist()
        # productIds = sorted(productIds)
        print('**********' , len(self.IngredientsInString))
        print('$$$$$' , len(productIds))
        print('###%%%' , len(featureSet))
        occurenceOfIngredientsInData = {}
        ctr =0;
        for i in range (len(self.IngredientsInString)):

            # if(ctr%100 ==0 ): print ('^^^^' ,ctr/100)
            if(ctr%557==0): print('ctr: ', ctr , 'productid ' , productIds[i], 'selfingredients' , self.IngredientsInString[i])
            wordCount = len(featureSet)*['0']
            try: #world finals B|
                blob = TextBlob(self.IngredientsInString[i])
                blob = blob.words
                blob = blob.singularize()
                blob = sorted(blob)
                for k in blob:
                    if(k in featureSet):
                        index = featureSet.index(k)
                        wordCount[index] = 1
                occurenceOfIngredientsInData.update({productIds[i]:wordCount})
            except Exception as e:
                    continue









            # try:
            #     wordCount = len(featureSet)*['0']
            #     for j in range (len(featureSet)):
            #         blob = TextBlob(self.IngredientsInString[i])
            #         blob = blob.words
            #         blob = blob.singularize()
            #         blob = sorted(blob)
            #         if(featureSet[j] in blob):
            #             wordCount.append(1)
            #         else: wordCount.append(0)
            #     occurenceOfIngredientsInData.update({productIds[i]:wordCount})
            # except Exception as e:
            #     continue
            ctr+=1
        toFrame = pd.DataFrame(occurenceOfIngredientsInData).transpose()
        toFrame.columns = self.SetUpFeatures()
        toFrame.index.name = 'ProductIDs'
        toFrame['ProductIDs'] = toFrame.index
        return toFrame


    
    def Ingredients(self, Ingredient):
        vals = re.findall(r'\(E\d+\)|([^\W\d]+(?:\s+[^\W\d]+)*)', Ingredient)
        ingredientsList = [x for x in vals if x]
        return ingredientsList

    def getAllergenInformation(self,gtin):
        df_code = self.df_AllergenCodes
        tag = df_code['Keywords'].tolist()
        code = df_code['Code'].tolist()

        df_allergen = self.df_AllergenData
        productId = df_allergen['Gtin'].astype(str).tolist()
        ingredientsInfo = df_allergen['IngredientsInfo'].tolist()
        languageInfo = df_allergen['LanguageInfo'].tolist()

        df_translatedItems = self.df_Translated
        originalItem = df_translatedItems['Word'].tolist()
        translation = df_translatedItems['Translation'].tolist()

        df_Info = self.df_AllergenInformation
        Gtin_code = df_Info['Gtin'].astype(str).tolist()
        code_value_0 = df_Info['Allergen[0]-Type-Value'].tolist()
        code_value_1 = df_Info['Allergen[1]-Type-Value'].tolist()
        code_value_2 = df_Info['Allergen[2]-Type-Value'].tolist()
        code_value_3 = df_Info['Allergen[3]-Type-Value'].tolist()
        code_value_4 = df_Info['Allergen[4]-Type-Value'].tolist()
        code_value_5 = df_Info['Allergen[5]-Type-Value'].tolist()
        code_value_6 = df_Info['Allergen[6]-Type-Value'].tolist()
        code_value_7 = df_Info['Allergen[7]-Type-Value'].tolist()
        code_value_8 = df_Info['Allergen[8]-Type-Value'].tolist()
        code_value_9 = df_Info['Allergen[9]-Type-Value'].tolist()
        code_value_10 = df_Info['Allergen[10]-Type-Value'].tolist()
        code_value_11 = df_Info['Allergen[11]-Type-Value'].tolist()
        code_value_12 = df_Info['Allergen[12]-Type-Value'].tolist()
        code_value_13 = df_Info['Allergen[13]-Type-Value'].tolist()
        code_value_14 = df_Info['Allergen[14]-Type-Value'].tolist()
        code_value_15 = df_Info['Allergen[15]-Type-Value'].tolist()
        code_value_16 = df_Info['Allergen[16]-Type-Value'].tolist()

        containment_value_0 = df_Info['Allergen[0]-Containment-Value'].tolist()
        containment_value_1 = df_Info['Allergen[1]-Containment-Value'].tolist()
        containment_value_2 = df_Info['Allergen[2]-Containment-Value'].tolist()
        containment_value_3 = df_Info['Allergen[3]-Containment-Value'].tolist()
        containment_value_4 = df_Info['Allergen[4]-Containment-Value'].tolist()
        containment_value_5 = df_Info['Allergen[5]-Containment-Value'].tolist()
        containment_value_6 = df_Info['Allergen[6]-Containment-Value'].tolist()
        containment_value_7 = df_Info['Allergen[7]-Containment-Value'].tolist()
        containment_value_8 = df_Info['Allergen[8]-Containment-Value'].tolist()
        containment_value_9 = df_Info['Allergen[9]-Containment-Value'].tolist()
        containment_value_10 = df_Info['Allergen[10]-Containment-Value'].tolist()
        containment_value_11 = df_Info['Allergen[11]-Containment-Value'].tolist()
        containment_value_12 = df_Info['Allergen[12]-Containment-Value'].tolist()
        containment_value_13 = df_Info['Allergen[13]-Containment-Value'].tolist()
        containment_value_14 = df_Info['Allergen[14]-Containment-Value'].tolist()
        containment_value_15 = df_Info['Allergen[15]-Containment-Value'].tolist()
        containment_value_16 = df_Info['Allergen[16]-Containment-Value'].tolist()

        try:
            indexOfItem = Gtin_code.index(gtin)
        except Exception as e:
            pass

        containsAllergen = [(code_value_0[indexOfItem], containment_value_0[indexOfItem]),
                            (code_value_1[indexOfItem], containment_value_1[indexOfItem]),
                            (code_value_2[indexOfItem], containment_value_2[indexOfItem]),
                            (code_value_3[indexOfItem], containment_value_3[indexOfItem]),
                            (code_value_4[indexOfItem], containment_value_4[indexOfItem]),
                            (code_value_5[indexOfItem], containment_value_5[indexOfItem]),
                            (code_value_6[indexOfItem], containment_value_6[indexOfItem]),
                            (code_value_7[indexOfItem], containment_value_7[indexOfItem]),
                            (code_value_8[indexOfItem], containment_value_8[indexOfItem]),
                            (code_value_9[indexOfItem], containment_value_9[indexOfItem]),
                            (code_value_10[indexOfItem], containment_value_10[indexOfItem]),
                            (code_value_11[indexOfItem], containment_value_11[indexOfItem]),
                            (code_value_12[indexOfItem], containment_value_12[indexOfItem]),
                            (code_value_13[indexOfItem], containment_value_13[indexOfItem]),
                            (code_value_14[indexOfItem], containment_value_14[indexOfItem]),
                            (code_value_15[indexOfItem], containment_value_15[indexOfItem]),
                            (code_value_16[indexOfItem], containment_value_16[indexOfItem])]

        containsAllergen = list(set([(x, y) for x, y in containsAllergen if (str(x) != ('nan') and str(y) != 'nan' and str(y) != 'FREE_FROM')]))
        cleanedListContainsAllergen = set([x for x in containsAllergen if str(x) != 'nan'])
        cleanedListContainsAllergen = [x for x, y in cleanedListContainsAllergen]
        cleanedListContainsAllergen = list(cleanedListContainsAllergen)

        return cleanedListContainsAllergen
    
  


    def getPreservativesInfo(self, gtin):
        df_allergen = self.df_AllergenData
        ingredients = df_allergen['IngredientsInfo'].tolist()
        GtinList = df_allergen['Gtin'].astype(str).tolist()
        languageInfo = df_allergen['LanguageInfo'].tolist()

        df_preservativesInfo = self.df_preservativesInfo
        codeOfPreservative = df_preservativesInfo['Code'].tolist()
        keywords = df_preservativesInfo['Keywords'].tolist()

        df_translatedItems = self.df_Translated
        originalItem = df_translatedItems['Word'].tolist()
        translation = df_translatedItems['Translation'].tolist()

        
        productId = GtinList.index(gtin)
        code_value_0 = df_allergen['Additive[0]-Type-Value'].tolist()
        code_value_1 = df_allergen['Additive[1]-Type-Value'].tolist()
        code_value_2 = df_allergen['Additive[2]-Type-Value'].tolist()
        code_value_3 = df_allergen['Additive[3]-Type-Value'].tolist()
        code_value_4 = df_allergen['Additive[4]-Type-Value'].tolist()
        code_value_5 = df_allergen['Additive[5]-Type-Value'].tolist()
        code_value_6 = df_allergen['Additive[6]-Type-Value'].tolist()
        code_value_7 = df_allergen['Additive[7]-Type-Value'].tolist()
        code_value_8 = df_allergen['Additive[8]-Type-Value'].tolist()
        code_value_9 = df_allergen['Additive[9]-Type-Value'].tolist()
        code_value_10 = df_allergen['Additive[10]-Type-Value'].tolist()
        code_value_11 = df_allergen['Additive[11]-Type-Value'].tolist()
        code_value_12 = df_allergen['Additive[12]-Type-Value'].tolist()
        code_value_13 = df_allergen['Additive[13]-Type-Value'].tolist()
        code_value_14 = df_allergen['Additive[14]-Type-Value'].tolist()
        code_value_15 = df_allergen['Additive[15]-Type-Value'].tolist()
        code_value_16 = df_allergen['Value_001'].tolist()
        code_value_17 = df_allergen['Value_002'].tolist()
        code_value_18 = df_allergen['Value_003'].tolist()

        containsAllergen = [code_value_0[productId], code_value_1[productId], code_value_2[productId],
                            code_value_3[productId], code_value_4[productId],
                            code_value_5[productId], code_value_6[productId], code_value_7[productId],
                            code_value_8[productId], code_value_9[productId],
                            code_value_10[productId], code_value_11[productId], code_value_12[productId],
                            code_value_13[productId], code_value_14[productId],
                            code_value_15[productId], code_value_16[productId], code_value_17[productId],
                            code_value_18[productId]]

        cleanedAllergenList = np.array(containsAllergen)

        if('nan' in cleanedAllergenList): cleanedAllergenList = [x for x in cleanedAllergenList if x != 'nan']
        else: cleanedAllergenList = [x for x in cleanedAllergenList if not math.isnan(float(x))]

        # setOfAllergenInIngredients = set(cleanedList)
        setOfAllergenInData = set(cleanedAllergenList)

        # InIngredientsButNotInData = list(setOfAllergenInIngredients - setOfAllergenInData)
        # InDataButNotInIngredients = list(setOfAllergenInData - setOfAllergenInIngredients)
        return list(setOfAllergenInData)



    
   

    def PrintReport(self):
        gtinList = self.df_GtinList
        X = self.SetUpDataFrameFromFeatures()
        check = X['ProductIDs'].astype(str).tolist()
        check = sorted(check)
        print('+++++' , len(check))
        
        productIds = X['ProductIDs'].astype(str).tolist()
        productIds = sorted(productIds)
        print('ttttttttt' , len(productIds))
        AllergenDBList = []
        PreservativeDBList = []
        dictToBePrinted = {}
        # len(productIds)
        codecompletion = 0
        X = X.as_matrix()
        for j in range (len(productIds)):
            allergensToBePrinted = []
            preservativesToBePrinted = []
            
            
            
            # if(productIds[j] in check): continue


            allergen = self.getAllergenInformation(productIds[j])
            preservatives = self.getPreservativesInfo(productIds[j])
            # print("hhhhh" , productIds[j] ,   preservatives) #indata
            # print("kkkkk" , productIds[j] ,   preservatives[1]) #expected val
            # print(productIds[j]  ,   allergen)
            sortedAllergenInDatabaase = (allergen)

            # sortedAllergenShouldBe = sorted(allergen[1])
            sortedPreservativesInDatabase = (preservatives)
            # sortedPreservativesShouldBe = sorted(preservatives[1])
            

            fixedListForAllergensInData = ['']*17 # It has max capacity of 17 elements from the database
            # fixedListForAllergensMyList = [''] * 17
            fixedListForPreservativesInData = [''] * 19  # It has max capacity of 19 elements from the database
            # fixedListForPreservativesMyList = [''] * 19
            
            #### SAME HAS TO BE DONE FOR ALL THREE TYPES
            if bool (sortedAllergenInDatabaase):
                for i in range(len(sortedAllergenInDatabaase)) :
                    fixedListForAllergensInData[i] = sortedAllergenInDatabaase[i]


            # if(sortedAllergenInDatabaase == sortedAllergenShouldBe):
            #   allergensToBePrinted.append(['SAME'] + ['']*17 + ['']*17 )
            # else:

            #print(sortedAllergenInDatabaase)
            AllergenDBList.append(sortedAllergenInDatabaase)
            PreservativeDBList.append(sortedPreservativesInDatabase)
            allergensToBePrinted.append(['NOT SAME'] + fixedListForAllergensInData + ['']*17 ) 
            #### SAME HAS TO BE DONE

            #print(allergensToBePrinted)
            if bool (sortedPreservativesInDatabase):
                for i in range(len(sortedPreservativesInDatabase)) :
                    fixedListForPreservativesInData[i] = sortedPreservativesInDatabase[i]

            # if  bool(fixedListForPreservativesMyList):
            #     for i in range(len(sortedPreservativesShouldBe)) :
            #         fixedListForPreservativesMyList[i] = sortedPreservativesShouldBe[i]


            # if(sortedPreservativesInDatabase == sortedPreservativesShouldBe):
            #     preservativesToBePrinted.append(['SAME'] + ['']*19 + ['']*19 )
            # else:
                preservativesToBePrinted.append(['NOT SAME'] + fixedListForPreservativesInData )



            finalListToBePrinted = allergensToBePrinted + preservativesToBePrinted
            # finalListToBePrinted = finalListToBePrinted[0] + finalListToBePrinted[1]

            if(j == 1000 or j == 2000 or j == 3000 or j == 4000):
                codecompletion += 10
                print('Process completed:', codecompletion, '%')
            # print(j,'-----',productIds[j])
            dictToBePrinted.update({productIds[j]: finalListToBePrinted})

        mlb_allergen = MultiLabelBinarizer()
        #print(mlb_allergen.fit_transform(AllergenDBList))
        Y_allergen = mlb_allergen.fit_transform(AllergenDBList)
        print (mlb_allergen.classes_)
       # np.insert(Y,mlb_allergen.classes_,axis=1)

        ZZ=pd.DataFrame(Y_allergen,columns = list(mlb_allergen.classes_))
        ZZ.insert(0, 'ID', productIds, allow_duplicates=False)
        ZZ.to_csv('File.csv', sep=',', encoding='latin-1', index = False)
        #above this X have it on form of a matrix
       
        print('X' , len(X) , '    Y' , len(Y_allergen) , '     Tot' , len(AllergenDBList))
        X_train, X_test, y_train, y_test = train_test_split(X, Y_allergen, test_size=0.3, random_state=42)
        model_6 = KNeighborsClassifier(n_neighbors=10,algorithm='auto')
        model_6.fit(X_train, y_train)
        test = tree.DecisionTreeClassifier()
        test.fit(X_train, y_train)
        clf = RandomForestClassifier(n_estimators=5)
        clf = clf.fit(X_train, y_train)
        # cart = DecisionTreeClassifier()
        # num_instances = len(X)
        # kfold = cross_validation.KFold(n=num_instances, n_folds=10, random_state=7)
        # num_trees = 100
        # model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=7)
        # results = cross_validation.cross_val_score(model, X, Y_allergen, cv=kfold)
        # print(results.mean())
        print('KNN: allergen', accuracy_score(y_test, model_6.predict(X_test)))
        print('Tree: allergen', accuracy_score(y_test, test.predict(X_test)))
        print('Tree: allergen', accuracy_score(y_test, clf.predict(X_test)))
        # print('bag: allergen', accuracy_score(y_test, bagging.predict(X_test)))

        mlb_preservative = MultiLabelBinarizer()
        Y_preservative = mlb_preservative.fit_transform(PreservativeDBList)
        
        file2=pd.DataFrame(Y_preservative,columns = list(mlb_preservative.classes_))
        file2.insert(0, 'ID', productIds, allow_duplicates=False)

        file2.to_csv('File2.csv', sep=',', encoding='latin-1', index = False)


        print (mlb_preservative.classes_)
        print('X' , len(X) , '    Y' , len(Y_preservative) , '     Tot' , len(PreservativeDBList))
        X_train, X_test, y_traine, y_teste = train_test_split(X, Y_preservative, test_size=0.3, random_state=42)
        model_7 = KNeighborsClassifier(algorithm='auto')
        model_7.fit(X_train, y_traine)
        tester = tree.DecisionTreeClassifier()
        tester.fit(X_train, y_traine)
        clfe = RandomForestClassifier(n_estimators=5)
        clfe = clfe.fit(X_train, y_traine)
        print('tree: preservative ', accuracy_score(y_teste, tester.predict(X_test)))
        print('KNN: preservative ', accuracy_score(y_teste, model_7.predict(X_test)))
        print('Tree: allergen', accuracy_score(y_teste, clfe.predict(X_test)))







        #
        # columnHeaders = self.df_ColumnHeaders
        # columnNames = [columnHeaders['ColumnHeader1'].tolist() ,columnHeaders['ColumnHeader2'].tolist() , columnHeaders['ColumnHeader3'].tolist()]

        columnNames =['AllergenOutput', 'AllergenInDatabase1',  'AllergenInDatabase2',  'AllergenInDatabase3',  'AllergenInDatabase4',
                       'AllergenInDatabase5',   'AllergenInDatabase6',  'AllergenInDatabase7',  'AllergenInDatabase8',  'AllergenInDatabase9',
                       'AllergenInDatabase10',  'AllergenInDatabase11', 'AllergenInDatabase12', 'AllergenInDatabase13', 'AllergenInDatabase14',
                       'AllergenInDatabase15',  'AllergenInDatabase16', 'AllergenInDatabase17', 'AllergenExpected1',    'AllergenExpected2',
                       'AllergenExpected3', 'AllergenExpected4',    'AllergenExpected5',    'AllergenExpected6',    'AllergenExpected7',
                       'AllergenExpected8', 'AllergenExpected9',    'AllergenExpected10',   'AllergenExpected11',   'AllergenExpected12',
                       'AllergenExpected13',    'AllergenExpected14',   'AllergenExpected15',   'AllergenExpected16',   'AllergenExpected17',
                        'PreservativesOutput',  'PreservativesInDatabase1', 'PreservativesInDatabase2', 'PreservativesInDatabase3',
                       'PreservativesInDatabase4',  'PreservativesInDatabase5', 'PreservativesInDatabase6', 'PreservativesInDatabase7',
                       'PreservativesInDatabase8',  'PreservativesInDatabase9', 'PreservativesInDatabase10',    'PreservativesInDatabase11',
                       'PreservativesInDatabase12', 'PreservativesInDatabase13',    'PreservativesInDatabase14',    'PreservativesInDatabase15',
                       'PreservativesInDatabase16', 'PreservativesInDatabase17',    'PreservativesInDatabase18',    'PreservativesInDatabase19',
                       ]

        # toFrame = pd.DataFrame(dictToBePrinted).transpose()
        # toFrame.columns = columnNames
        # toFrame.index.name = 'ProductIds'
        # toFrame['ProductIDs'] = toFrame.index

       
        return 




def Main():
    warnings.filterwarnings('ignore')
    correct = Correction()
    APPWFrame = correct.PrintReport()
    print('Process completed: 50%')
    
    print('Process completed: 75%')
    
    # mergedFrame = pd.merge(intrastatFrame, nutritionFrame, on='ProductIDs', how='outer')
    # mergedFrameTemp = pd.merge(APPWFrame, mergedFrame, on='ProductIDs', how='outer')
    # mergedFrameTemp = pd.merge(mergedFrameTemp, MPGFrame, on='ProductIDs', how='outer')

    # mergedFrameTemp = APPWFrame.reindex_axis(['ProductIDs'] + list([a for a in APPWFrame.columns if a != 'ProductIDs']), axis=1)
    # mergedFrameTemp['ProductIDs'] = ['XXX' + str(int(x)) for x in mergedFrameTemp['ProductIDs']]
    # mergedFrameTemp['ProductIDs'] = mergedFrameTemp['ProductIDs'].astype(str)
    # mergedFrameTemp.to_csv('ResultsFile.csv', sep=',', encoding='latin-1', index = False)


if __name__ == '__main__':
    print('The process is running. This may take a few minutes ...')
    print('For i5 processor and 8GB RAM it might take around 20-25 minutes ...')
    print('Start Time: ', datetime.now())
    Main()
    print('Done!')
    print('End Time: ', datetime.now())
    print('The results can be seen in ResultsFile.csv which is created in the same folder')
