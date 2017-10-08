# -*- coding: utf-8 -*-
import os, sys

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
        self.df_AllergenCodes = pd.read_csv('./DataFilesForDataProcessing/AleergenTypeCode.csv', encoding='latin-1')
        self.df_AllergenInformation = pd.read_csv('./DataFilesForDataProcessing/AllergenInformation.csv', encoding='latin-1')
        self.IngredientsFrame = pd.read_csv('./DataFilesForDataProcessing/AllegenData.csv', encoding= 'latin-1')

        self.df_preservativesInfo = pd.read_csv('./DataFilesForDataProcessing/AdditiveInformation.csv', encoding='latin-1')
        self.df_Translated = pd.read_csv('./DataFilesForDataProcessing/frame.csv', encoding='latin-1')
        self.IngredientsInString = self.IngredientsFrame['IngredientsInfo'].tolist()
        self.languages=self.IngredientsFrame['LanguageInfo'].tolist()
        self.temp=(self.IngredientsFrame['Gtin'].tolist())
        
        self.df_GtinList = pd.read_csv('./DataFilesForDataProcessing/GtinList.csv', encoding='latin-1')
        
        self.df_ColumnHeaders = pd.read_csv('./DataFilesForDataProcessing/ColumnHeaders.csv', encoding='latin-1')



    def SetUpFeatures(self):
        featuresSet = set()
        for i in range (len(self.IngredientsInString)):
            try:
                vals = re.findall(ur'\(?\bE\s*\d+\w*\)?\W*|\b([^\W\d]+(?:\s+[^\W\d]+)*)\b', self.IngredientsInString[i], re.UNICODE)
                self.ListOfIngredients = [x for x in vals if x]
                for j in range (len(self.ListOfIngredients)):
                    if(len(self.ListOfIngredients[j]) < 4): continue
                    featuresSet.add(self.ListOfIngredients[j].lower())
            except Exception as e:
                continue
        return featuresSet


    def SetUpDataFrameFromFeatures(self):
        gtinList = self.df_GtinList
        featureSet = list(self.SetUpFeatures())
        # productIds = self.IngredientsFrame['Gtin'].astype(str).tolist()
        productIds = gtinList['ProductIDs'].astype(str).tolist()
        productIds = sorted(productIds)
        print('**********' , len(self.IngredientsInString))
        occurenceOfIngredientsInData = {}
        for i in range (len(self.IngredientsInString)):
            try:
                wordCount = []
                for j in range (len(featureSet)):
                    if(featureSet[j] in self.IngredientsInString[i].lower()):
                        wordCount.append(1)
                    else: wordCount.append(0)
                occurenceOfIngredientsInData.update({productIds[i]:wordCount})
            except Exception as e:
                continue
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

        # try:
        #     stringOfIngredients = ingredientsInfo[productId.index(gtin)]
        #     listOfIngredients = self.Ingredients(stringOfIngredients)
        # except Exception as e:
        #     pass
        #     return

        # translatedListOfItems = []
        # allergiesShouldContain = []
        # language = languageInfo[productId.index(gtin)]
        # if (language != 'en'):
        #     for item in listOfIngredients:
        #         try:
        #             item = item.lower()
        #             if (len(item) < 3): continue
        #             index = originalItem.index(item)
        #             translatedItem = translation[index]
        #             translatedListOfItems.append(translatedItem.lower())
        #         except Exception as e:
        #             pass
        #     for i in range(len(translatedListOfItems)):
        #         for j in range(len(tag)):
        #             if (tag[j] in translatedListOfItems[i].lower()):
        #                 allergiesShouldContain.append(code[j].rstrip())
        # else:
        #     for i in range(len(listOfIngredients)):
        #         for j in range(len(tag)):
        #             if (len(listOfIngredients[i])<3) : continue
        #             if (tag[j] in listOfIngredients[i].lower()):
        #                 allergiesShouldContain.append( code[j].rstrip())
        # uniqueElementsShouldContain = list(set(allergiesShouldContain))
        # InDataButNotInIngredients = list(cleanedListContainsAllergen - uniqueElementsShouldContain)
        # InIngredientsButNotInData = list(uniqueElementsShouldContain - cleanedListContainsAllergen)

        # Commented return proides the difference in the datasets that
        # which contains originally and which should contain according to our recommendation

        # return InDataButNotInIngredients, InIngredientsButNotInData
        return cleanedListContainsAllergen
    
    def getAllIngredients(self):
        df = self.df_AllergenData
        ingredients = df['IngredientsInfo'].tolist()
        productId = df['Gtin'].tolist()
        languageInfo = df['LanguageInfo'].tolist()

        wordToTranslation = {}
        uniqueIngredients = set()
        # product_ingredients_mapping = defaultdict(list)
        x = 0
        for i in range(len(ingredients)):
            try:
                if (languageInfo[i] != 'en'):
                    vals = re.findall(r'\(E\d+\)|([^\W\d]+(?:\s+[^\W\d]+)*)', ingredients[i])
                    instance = Translate()
                    for item in vals:
                        if (len(item) < 3): continue
                        translatedItem = instance.TranslateWord(item, languageInfo[i])
                        uniqueIngredients.add(translatedItem.lower())
                        wordToTranslation.update({item: [translatedItem, instance.DetectLanguage(translatedItem)]})
                else:
                    vals = re.findall(r'\(E\d+\)|([^\W\d]+(?:\s+[^\W\d]+)*)', ingredients[i])
                    for j in vals:
                        if (len(item) < 3): continue
                        uniqueIngredients.add(j.lower())
                        # product_ingredients_mapping[j].append(str(productId[i]))
                        wordToTranslation.update({item: [item, 'en']})
            except Exception as e:
                x += 1

        # Dataframe of items ===> productId mapping
        # frame = pd.DataFrame(dict([ (k,Series(v)) for k,v in product_ingredients_mapping.items() ])).transpose()
        frame = pd.DataFrame(wordToTranslation, index=['Translation', 'Language']).transpose()
        frame.to_csv('frame.csv', sep=',', encoding='latin-1')
        return uniqueIngredients



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

        try:
            productId = GtinList.index(gtin)
            ingredientsOfProduct = ingredients[productId]
            rrequiredPreservatives = re.findall(r'E+[0-9]..\w|E+[0-9]..|E+ [0-9]..\w|E+ [0-9]..', ingredientsOfProduct)
            cleanedList = []
            for preservative in rrequiredPreservatives:
                cleanedList.append(preservative.replace(" ", ""))

            stringOfIngredients = ingredients[GtinList.index(gtin)]
            listOfIngredients = self.Ingredients(stringOfIngredients)

            translatedListOfItems = []
            preservativesShouldContain = []
            language = languageInfo[GtinList.index(gtin)]
            if (language != 'en'):
                for item in listOfIngredients:
                    try:
                        item = item.lower()
                        if (len(item) < 3): continue
                        index = originalItem.index(item)
                        translatedItem = translation[index]
                        translatedListOfItems.append(translatedItem.lower())
                    except Exception as e:
                        pass
                for i in range(len(translatedListOfItems)):
                    for j in range(len(keywords)):
                        if (keywords[j] in translatedListOfItems[i].lower()):
                            preservativesShouldContain.append(codeOfPreservative[j].rstrip())
            else:
                for i in range(len(listOfIngredients)):
                    for j in range(len(keywords)):
                        if (str(keywords[j]) in str(listOfIngredients[i]).lower()):
                            preservativesShouldContain.append(codeOfPreservative[j].rstrip())
            cleanedList = cleanedList + preservativesShouldContain
            cleanedList = [x for x in cleanedList if x != 'nan']
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            pass

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

        setOfAllergenInIngredients = set(cleanedList)
        setOfAllergenInData = set(cleanedAllergenList)

        # InIngredientsButNotInData = list(setOfAllergenInIngredients - setOfAllergenInData)
        # InDataButNotInIngredients = list(setOfAllergenInData - setOfAllergenInIngredients)
        return list(setOfAllergenInData), list(setOfAllergenInIngredients)



    
   

    def PrintReport(self):
        gtinList = self.df_GtinList
        X = self.SetUpDataFrameFromFeatures()
        check = X['ProductIDs'].astype(str).tolist()
        check = sorted(check)
        print('+++++' , len(check))
       
        productIds = X['ProductIDs'].astype(str).tolist()
        productIds = sorted(productIds)
        print('ttttttttt' , len(productIds))
        total = []

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
            
            # print(productIds[j]  ,   allergen)
            sortedAllergenInDatabaase = sorted(allergen)
            # sortedAllergenShouldBe = sorted(allergen[1])
            sortedPreservativesInDatabase = sorted(preservatives[0])
            sortedPreservativesShouldBe = sorted(preservatives[1])
            

            fixedListForAllergensInData = ['']*17 # It has max capacity of 17 elements from the database
            fixedListForAllergensMyList = [''] * 17
            fixedListForPreservativesInData = [''] * 19  # It has max capacity of 19 elements from the database
            fixedListForPreservativesMyList = [''] * 19
            
            #### SAME HAS TO BE DONE FOR ALL THREE TYPES
            if bool (sortedAllergenInDatabaase):
                for i in range(len(sortedAllergenInDatabaase)) :
                    fixedListForAllergensInData[i] = sortedAllergenInDatabaase[i]


            # if(sortedAllergenInDatabaase == sortedAllergenShouldBe):
            #   allergensToBePrinted.append(['SAME'] + ['']*17 + ['']*17 )
            # else:

           
            #print(sortedAllergenInDatabaase)







            total.append(sortedAllergenInDatabaase)

            allergensToBePrinted.append(['NOT SAME'] + fixedListForAllergensInData + ['']*17 ) 
            #### SAME HAS TO BE DONE

            #print(allergensToBePrinted)





            if bool (sortedPreservativesInDatabase):
                for i in range(len(sortedPreservativesInDatabase)) :
                    fixedListForPreservativesInData[i] = sortedPreservativesInDatabase[i]

            if  bool(fixedListForPreservativesMyList):
                for i in range(len(sortedPreservativesShouldBe)) :
                    fixedListForPreservativesMyList[i] = sortedPreservativesShouldBe[i]


            if(sortedPreservativesInDatabase == sortedPreservativesShouldBe):
                preservativesToBePrinted.append(['SAME'] + ['']*19 + ['']*19 )
            else:
                preservativesToBePrinted.append(['NOT SAME'] + fixedListForPreservativesInData + fixedListForPreservativesMyList )



            finalListToBePrinted = allergensToBePrinted + preservativesToBePrinted
            finalListToBePrinted = finalListToBePrinted[0] + finalListToBePrinted[1]

            if(j == 1000 or j == 2000 or j == 3000 or j == 4000):
                codecompletion += 10
                print('Process completed:', codecompletion, '%')
            # print(j,'-----',productIds[j])
            dictToBePrinted.update({productIds[j]: finalListToBePrinted})

        mlb = MultiLabelBinarizer()
        #print(mlb.fit_transform(total))
        Y = mlb.fit_transform(total)

       



        #above this X have it on form of a matrix
       
        print('X' , len(X) , '    Y' , len(Y) , '     Tot' , len(total))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        model_6 = KNeighborsClassifier()
        model_6.fit(X_train, y_train)



        print('KNN: ', accuracy_score(y_test, model_6.predict(X_test)))
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
                       'PreservativesExpected1',    'PreservativesExpected2',   'PreservativesExpected3',   'PreservativesExpected4',
                       'PreservativesExpected5',    'PreservativesExpected6',   'PreservativesExpected7',   'PreservativesExpected8',
                       'PreservativesExpected9',    'PreservativesExpected10',  'PreservativesExpected11',  'PreservativesExpected12',
                       'PreservativesExpected13',   'PreservativesExpected14',  'PreservativesExpected15',  'PreservativesExpected16',
                       'PreservativesExpected17',   'PreservativesExpected18',  'PreservativesExpected19', ]

        toFrame = pd.DataFrame(dictToBePrinted).transpose()
        toFrame.columns = columnNames
        toFrame.index.name = 'ProductIds'
        toFrame['ProductIDs'] = toFrame.index
        # toFrame.to_csv('OutputFile.csv', sep=',', encoding='latin-1')
        return toFrame




def Main():
    warnings.filterwarnings('ignore')
    correct = Correction()
    APPWFrame = correct.PrintReport()
    print('Process completed: 50%')
    
    print('Process completed: 75%')
    
    # mergedFrame = pd.merge(intrastatFrame, nutritionFrame, on='ProductIDs', how='outer')
    # mergedFrameTemp = pd.merge(APPWFrame, mergedFrame, on='ProductIDs', how='outer')
    # mergedFrameTemp = pd.merge(mergedFrameTemp, MPGFrame, on='ProductIDs', how='outer')

    mergedFrameTemp = APPWFrame.reindex_axis(['ProductIDs'] + list([a for a in APPWFrame.columns if a != 'ProductIDs']), axis=1)
    # mergedFrameTemp['ProductIDs'] = ['XXX' + str(int(x)) for x in mergedFrameTemp['ProductIDs']]
    # mergedFrameTemp['ProductIDs'] = mergedFrameTemp['ProductIDs'].astype(str)
    mergedFrameTemp.to_csv('ResultsFile.csv', sep=',', encoding='latin-1', index = False)


if __name__ == '__main__':
    print('The process is running. This may take a few minutes ...')
    print('For i5 processor and 8GB RAM it might take around 20-25 minutes ...')
    print('Start Time: ', datetime.now())
    Main()
    print('Done!')
    print('End Time: ', datetime.now())
    print('The results can be seen in ResultsFile.csv which is created in the same folder')