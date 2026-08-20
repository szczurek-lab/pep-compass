from __future__ import print_function
import argparse
import warnings
import subprocess
import itertools
from collections import Counter
import pickle
import uuid
from time import sleep
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import zipfile
import getopt
import sys
import os
import numpy as np
import pandas as pd
import math
from itertools import repeat
import csv
import re
import glob
import time
from argparse import RawTextHelpFormatter
import uuid
import warnings
import os
import random
import shutil
import tempfile
nf_path = os.path.dirname(__file__)

# Load selected feature lists if available. If the DATA folder is missing,
# avoid failing import — set placeholders and allow predictor to operate in
# fallback (model-only) mode.
try:
    aa = pd.read_csv(os.path.join(nf_path, 'Data', 'selected_features_mrmr1000_new.csv'))
    filtered_aa = aa[aa['SelectedFeatures'].str.startswith('TPC')]
    selected_tripeptides = filtered_aa['SelectedFeatures'].tolist()

    filtered_bb = aa[aa['SelectedFeatures'].str.startswith('DPC')]
    selected_dipeptides = filtered_bb['SelectedFeatures'].tolist()
    EIPRED_DATA_PRESENT = True
except FileNotFoundError:
    # Data files not present — continue with placeholders and let PredictorEIPred
    # either run the full pipeline later (if files appear) or fall back to
    # model-only inference.
    selected_tripeptides = []
    selected_dipeptides = []
    EIPRED_DATA_PRESENT = False


def features_calulate(file_path, output_file):
    # Standard amino acid codes
    std = "ACDEFGHIKLMNPQRSTVWY"


    def aac_comp(file, out):
        filename, _ = os.path.splitext(file)
        f = open(out, 'w')
        sys.stdout = f
        df = pd.read_csv(file, header=None)
        zz = df.iloc[:, 0]
        print("AAC_A,AAC_C,AAC_D,AAC_E,AAC_F,AAC_G,AAC_H,AAC_I,AAC_K,AAC_L,AAC_M,AAC_N,AAC_P,AAC_Q,AAC_R,AAC_S,AAC_T,AAC_V,AAC_W,AAC_Y,")
        for j in zz:
            for i in std:
                count = 0
                for k in j:
                    temp1 = k
                    if temp1 == i:
                        count += 1
                composition = (count / len(j)) * 100
                print("%.2f" % composition, end=",")
            print("")
        f.truncate()


    def dpc_comp(file, q, out, selected_dipeptides):
        # Open the input file and read the data
        df = pd.read_csv(file, header=None)
        df1 = pd.DataFrame(df[0].str.upper())
        zz = df1.iloc[:, 0]

        # Open the output file
        with open(out, 'w') as f:
            # Redirect stdout to the output file
            original_stdout = sys.stdout
            sys.stdout = f

            # Print the header for selected dipeptides
            headers = [f"{dp}" for dp in selected_dipeptides]
            print(",".join(headers) + ",")

            # Calculate and print dipeptide composition for selected dipeptides
            for sequence in zz:
                compositions = []
                for dp in selected_dipeptides:
                    count = 0
                    for m3 in range(len(sequence) - q):
                        b = sequence[m3:m3 + q + 1:q]
                        if b == dp:
                            count += 1
                    composition = (count / (len(sequence) - q)) * 100
                    compositions.append(f"{composition:.2f}")
                print(",".join(compositions) + ",")

            # Restore original stdout
            sys.stdout = original_stdout

    def tpc_comp(file, out, selected_tripeptides):
        df = pd.read_csv(file, header=None)
        zz = df.iloc[:, 0]

        with open(out, 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f

            # Print header for selected tripeptides
            for tripeptide in selected_tripeptides:
                print(tripeptide, end=",")
            print("")

            # Calculate and print tripeptide composition for selected tripeptides
            for i in range(len(zz)):
                sequence = zz[i]
                seq_len = len(sequence)
                for tripeptide in selected_tripeptides:
                    count = 0
                    for m3 in range(seq_len - 2):
                        b = sequence[m3:m3 + 3]
                        if b == tripeptide:
                            count += 1
                    composition = (count / (seq_len - 2)) * 100
                    print(f"{composition:.2f}", end=',')
                print("")

            sys.stdout = original_stdout

    def atc(file,out):
        filename,file_ext = os.path.splitext(file)
        atom=pd.read_csv(nf_path+"/Data/atom.csv",header=None)
        at=pd.DataFrame()
        i = 0
        C_atom = []
        H_atom = []
        N_atom = []
        O_atom = []
        S_atom = []

        while i < len(atom):
            C_atom.append(atom.iloc[i,1].count("C"))
            H_atom.append(atom.iloc[i,1].count("H"))
            N_atom.append(atom.iloc[i,1].count("N"))
            O_atom.append(atom.iloc[i,1].count("O"))
            S_atom.append(atom.iloc[i,1].count("S"))
            i += 1
        atom["C_atom"]=C_atom
        atom["O_atom"]=O_atom
        atom["H_atom"]=H_atom
        atom["N_atom"]=N_atom
        atom["S_atom"]=S_atom
    ##############read file ##########
        test1 = pd.read_csv(file,header=None)
        dd = []
        for i in range(0, len(test1)):
            dd.append(test1[0][i].upper())
        test = pd.DataFrame(dd)
        count_C = 0
        count_H = 0
        count_N = 0
        count_O = 0
        count_S = 0
        count = 0
        i1 = 0
        j = 0
        k = 0
        C_ct = []
        H_ct = []
        N_ct = []
        O_ct = []
        S_ct = []
        while i1 < len(test) :
            while j < len(test[0][i1]) :
                while k < len(atom) :
                    if test.iloc[i1,0][j]==atom.iloc[k,0].replace(" ","") :
                        count_C = count_C + atom.iloc[k,2]
                        count_H = count_H + atom.iloc[k,3]
                        count_N = count_N + atom.iloc[k,4]
                        count_O = count_O + atom.iloc[k,5]
                        count_S = count_S + atom.iloc[k,6]
                    #count = count_C + count_H + count_S + count_N + count_O
                    k += 1
                k = 0
                j += 1
            C_ct.append(count_C)
            H_ct.append(count_H)
            N_ct.append(count_N)
            O_ct.append(count_O)
            S_ct.append(count_S)
            count_C = 0
            count_H = 0
            count_N = 0
            count_O = 0
            count_S = 0
            j = 0
            i1 += 1
        test["C_count"]=C_ct
        test["H_count"]=H_ct
        test["N_count"]=N_ct
        test["O_count"]=O_ct
        test["S_count"]=S_ct

        ct_total = []
        m = 0
        while m < len(test) :
            ct_total.append(test.iloc[m,1] + test.iloc[m,2] + test.iloc[m,3] + test.iloc[m,4] + test.iloc[m,5])
            m += 1
        test["count"]=ct_total
    ##########final output#####
        final = pd.DataFrame()
        n = 0
        p = 0
        C_p = []
        H_p = []
        N_p = []
        O_p = []
        S_p = []
        while n < len(test):
            C_p.append((test.iloc[n,1]/test.iloc[n,6])*100)
            H_p.append((test.iloc[n,2]/test.iloc[n,6])*100)
            N_p.append((test.iloc[n,3]/test.iloc[n,6])*100)
            O_p.append((test.iloc[n,4]/test.iloc[n,6])*100)
            S_p.append((test.iloc[n,5]/test.iloc[n,6])*100)
            n += 1
        final["ATC_C"] = C_p
        final["ATC_H"] = H_p
        final["ATC_N"] = N_p
        final["ATC_O"] = O_p
        final["ATC_S"] = S_p

        (final.round(2)).to_csv(out, index = None, encoding = 'utf-8')

    def bond(file,out) :
        tota = []
        hy = []
        Si = []
        Du = []
        b1 = []
        b2 = []
        b3 = []
        b4 = []
        bb = pd.DataFrame()
        filename, file_extension = os.path.splitext(file)
        df = pd.read_csv(file, header = None)
        bonds=pd.read_csv(nf_path+"/Data/bonds.csv", sep = ",")
        for i in range(0,len(df)) :
            tot = 0
            h = 0
            S = 0
            D = 0
            tota.append([i])
            hy.append([i])
            Si.append([i])
            Du.append([i])
            for j in range(0,len(df[0][i])) :
                temp = df[0][i][j]
                for k in range(0,len(bonds)) :
                    if bonds.iloc[:,0][k] == temp :
                        tot = tot + bonds.iloc[:,1][k]
                        h = h + bonds.iloc[:,2][k]
                        S = S + bonds.iloc[:,3][k]
                        D = D + bonds.iloc[:,4][k]
            tota[i].append(tot)
            hy[i].append(h)
            Si[i].append(S)
            Du[i].append(D)
        for m in range(0,len(df)) :
            b1.append(tota[m][1])
            b2.append(hy[m][1])
            b3.append(Si[m][1])
            b4.append(Du[m][1])

        bb["BTC_T"] = b1
        bb["BTC_H"] = b2
        bb["BTC_S"] = b3
        bb["BTC_D"] = b4

        bb.to_csv(out, index=None, encoding="utf-8")
    ############################PhysicoChemical Properties###################################
    import pandas as pd
    PCP= pd.read_csv(nf_path+'/Data/PhysicoChemical.csv', header=None)

    headers = ['PCP_PC','PCP_NC','PCP_NE','PCP_PO','PCP_NP','PCP_AL','PCP_CY','PCP_AR','PCP_AC','PCP_BS','PCP_NE_pH','PCP_HB','PCP_HL','PCP_NT','PCP_HX','PCP_SC','PCP_SS_HE','PCP_SS_ST','PCP_SS_CO','PCP_SA_BU','PCP_SA_EX','PCP_SA_IN','PCP_TN','PCP_SM','PCP_LR','PCP_Z1','PCP_Z2','PCP_Z3','PCP_Z4','PCP_Z5'];

    def encode(peptide):
        l=len(peptide);
        encoded=np.zeros(l);
        for i in range(l):
            if(peptide[i]=='A'):
                encoded[i] = 0;
            elif(peptide[i]=='C'):
                encoded[i] = 1;
            elif(peptide[i]=='D'):
                encoded[i] = 2;
            elif(peptide[i]=='E'):
                encoded[i] = 3;
            elif(peptide[i]=='F'):
                encoded[i] = 4;
            elif(peptide[i]=='G'):
                encoded[i] = 5;
            elif(peptide[i]=='H'):
                encoded[i] = 6;
            elif(peptide[i]=='I'):
                encoded[i] = 7;
            elif(peptide[i]=='K'):
                encoded[i] = 8;
            elif(peptide[i]=='L'):
                encoded[i] = 9;
            elif(peptide[i]=='M'):
                encoded[i] = 10;
            elif(peptide[i]=='N'):
                encoded[i] = 11;
            elif(peptide[i]=='P'):
                encoded[i] = 12;
            elif(peptide[i]=='Q'):
                encoded[i] = 13;
            elif(peptide[i]=='R'):
                encoded[i] = 14;
            elif(peptide[i]=='S'):
                encoded[i] = 15;
            elif(peptide[i]=='T'):
                encoded[i] = 16;
            elif(peptide[i]=='V'):
                encoded[i] = 17;
            elif(peptide[i]=='W'):
                encoded[i] = 18;
            elif(peptide[i]=='Y'):
                encoded[i] = 19;
            else:
                print('Wrong residue!');
        return encoded;
    def lookup(peptide,featureNum):
        l=len(peptide);
        peptide = list(peptide);
        out=np.zeros(l);
        peptide_num = encode(peptide);

        for i in range(l):
            out[i] = PCP[peptide_num[i]][featureNum];
        return sum(out);
    def pcp_1(file,out123):

        if(type(file) == str):
            seq = pd.read_csv(file,header=None);
            #seq=seq.T
            seq[0].values.tolist()
            seq=seq[0];
        else:
            seq  = file;

        l = len(seq);

        rows = PCP.shape[0]; # Number of features in our reference table
        col = 20 ; # Denotes the 20 amino acids

        seq=[seq[i].upper() for i in range(l)]
        sequenceFeature = [];
        sequenceFeature.append(headers); #To put property name in output csv

        for i in range(l): # Loop to iterate over each sequence
            nfeatures = rows;
            sequenceFeatureTemp = [];
            for j in range(nfeatures): #Loop to iterate over each feature
                featureVal = lookup(seq[i],j)
                if(len(seq[i])!=0):
                    sequenceFeatureTemp.append(round(featureVal/len(seq[i]),3));
                else:
                    sequenceFeatureTemp.append('NaN')

            sequenceFeature.append(sequenceFeatureTemp);

        out = pd.DataFrame(sequenceFeature);
        file = open(out123,'w')
        with file:
            writer = csv.writer(file);
            writer.writerows(sequenceFeature);
        return sequenceFeature;

    def DDOR(file,out) :
        df = pd.read_csv(file, header = None)
        df1 = pd.DataFrame(df[0].str.upper())
        f = open(out,'w')
        sys.stdout = f
        for i in std:
            print('DDR_'+i, end=",")
        print("")
        for i in range(0,len(df1)):
            s = df1[0][i]
            p = s[::-1]
            for j in std:
                zz = ([pos for pos, char in enumerate(s) if char == j])
                pp = ([pos for pos, char in enumerate(p) if char == j])
                ss = []
                for i in range(0,(len(zz)-1)):
                    ss.append(zz[i+1] - zz[i]-1)
                if zz == []:
                    ss = []
                else:
                    ss.insert(0,zz[0])
                    ss.insert(len(ss),pp[0])
                cc1=  (sum([e for e in ss])+1)
                cc = sum([e*e for e in ss])
                zz2 = cc/cc1
                print("%.2f"%zz2,end=",")
            print("")
        f.truncate()

    ##################################

    def entropy_single(seq):
        seq=seq.upper()
        num, length = Counter(seq), len(seq)
        return -sum( freq/length * math.log(freq/length, 2) for freq in num.values())

    def SE(filename,out):
        data=list((pd.read_csv(filename,sep=',',header=None)).iloc[:,0])
    #     print(data)
        Val=[]
        header=["SEP"]
        for i in range(len(data)):
            data1=''
            data1=str(data[i])
            data1=data1.upper()
            allowed = set(('A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'))
            is_data_invalid = set(data1).issubset(allowed)
            if is_data_invalid==False:
                print("Error: Please check for invalid inputs in the sequence.","\nError in: ","Sequence number=",i+1,",","Sequence = ",data[i],",","\nNOTE: Spaces, Special characters('[@_!#$%^&*()<>?/\\|}{~:]') and Extra characters(BJOUXZ) should not be there.")
                return
            Val.append(round((entropy_single(str(data[i]))),3))
            #print(Val[i])
            file= open(out,'w', newline='\n')#output file
            with file:
                writer=csv.writer(file,delimiter='\n');
                writer.writerow(header)
                writer.writerow(Val);
        return Val


    def SE_residue_level(filename,out):
        data=list((pd.read_csv(filename,sep=',',header=None)).iloc[:,0])
        data2=list((pd.read_csv(filename,sep=',',header=None)).iloc[:,0])
        Val=np.zeros(len(data))
        GH=[]
        for i in range(len(data)):
            my_list={'A':0,'C':0,'D':0,'E':0,'F':0,'G':0,'H':0,'I':0,'K':0,'L':0,'M':0,'N':0,'P':0,'Q':0,'R':0,'S':0,'T':0,'V':0,'W':0,'Y':0}
            data1=''
            data1=str(data[i])
            data1=data1.upper()
            allowed = set(('A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'))
            is_data_invalid = set(data1).issubset(allowed)
            if is_data_invalid==False:
                print("Error: Please check for invalid inputs in the sequence.","\nError in: ","Sequence number=",i+1,",","Sequence = ",data[i],",","\nNOTE: Spaces, Special characters('[@_!#$%^&*()<>?/\\|}{~:]') and Extra characters(BJOUXZ) should not be there.")
                return
            seq=data[i]
            seq=seq.upper()
            num, length = Counter(seq), len(seq)
            num=dict(sorted(num.items()))
            C=list(num.keys())
            F=list(num.values())
            for key, value in my_list.items():
                 for j in range(len(C)):
                    if key == C[j]:
                        my_list[key] = round(((F[j]/length)* math.log(F[j]/length, 2)),3)
            GH.append(list(my_list.values()))
        file= open(out,'w', newline='')#output file
        with file:
            writer=csv.writer(file);
            writer.writerow(('SER_A','SER_C','SER_D','SER_E','SER_F','SER_G','SER_H','SER_I','SER_K','SER_L','SER_M','SER_N','SER_P','SER_Q','SER_R','SER_S','SER_T','SER_V','SER_W','SER_Y'));
            writer.writerows(GH);
        return(GH)

    def RAAC(file,out):
        filename, file_extension = os.path.splitext(file)
        df = pd.read_csv(file, header = None)
        df1 = pd.DataFrame(df[0].str.upper())
        count = 0
        cc = []
        i = 0
        x = 0
        temp = pd.DataFrame()
        f = open(out,'w')
        sys.stdout = f
        print("RRI_A,RRI_C,RRI_D,RRI_E,RRI_F,RRI_G,RRI_H,RRI_I,RRI_K,RRI_L,RRI_M,RRI_N,RRI_P,RRI_Q,RRI_R,RRI_S,RRI_T,RRI_V,RRI_W,RRI_Y,")
        for q in range(0,len(df1)):
            while i < len(std):
                cc = []
                for j in df1[0][q]:
                    if j == std[i]:
                        count += 1
                        cc.append(count)
                    else:
                        count = 0
                while x < len(cc) :
                    if x+1 < len(cc) :
                        if cc[x]!=cc[x+1] :
                            if cc[x] < cc[x+1] :
                                cc[x]=0
                    x += 1
                cc1 = [e for e in cc if e!= 0]
                cc = [e*e for e in cc if e != 0]
                zz= sum(cc)
                zz1 = sum(cc1)
                if zz1 != 0:
                    zz2 = zz/zz1
                else:
                    zz2 = 0
                print("%.2f"%zz2,end=',')
                i += 1
            i = 0
            print(" ")
        f.truncate()

    PCP= pd.read_csv(nf_path+'/Data/PhysicoChemical.csv', header=None) #Our reference table for properties
    headers_1 = ['PRI_PC','PRI_NC','PRI_NE','PRI_PO','PRI_NP','PRI_AL','PRI_CY','PRI_AR','PRI_AC','PRI_BS','PRI_NE_pH','PRI_HB','PRI_HL','PRI_NT','PRI_HX','PRI_SC','PRI_SS_HE','PRI_SS_ST','PRI_SS_CO','PRI_SA_BU','PRI_SA_EX','PRI_SA_IN','PRI_TN','PRI_SM','PRI_LR']

    # Amino acid encoding dictionary
    amino_acid_dict = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
        'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
    }
    
    
    def encode(peptide):
        # Use a list comprehension for fast lookup
        encoded = np.array([amino_acid_dict.get(aa, -1) for aa in peptide])

        # Check for invalid residues
        if -1 in encoded:
            invalid_residues = [peptide[i] for i in range(len(peptide)) if encoded[i] == -1]
            for residue in invalid_residues:
                print(residue, 'is a wrong residue!')

        return encoded
    
    def lookup_1(peptide,featureNum):
        l=len(peptide);
        peptide = list(peptide);
        out=[];
        peptide_num = encode(peptide);

        for i in range(l):
            out.append(PCP[peptide_num[i]][featureNum]);
        return out;
    def binary_profile_1(file,featureNumb):
        if(type(file) == str):
            seq = pd.read_csv(file,header=None, sep=',');
            seq=seq.T
            seq[0].values.tolist()
            seq=seq[0];
        else:
            seq  = file;
        l = len(seq);
        bin_prof = [];
        for i in range(0,l):
            temp = lookup_1(seq[i],featureNumb);
            bin_prof.append(temp);
        return bin_prof;
    def repeats(file,out123):
        if(type(file) == str):
            seq = pd.read_csv(file,header=None, sep=',');
            #seq=seq.T
            seq[0].values.tolist()
            seq=seq[0];
        else:
            seq  = file;
        seq=[seq[i].upper() for i in range(len(seq))]
        dist =[];
        dist.append(headers_1);
        l = len(seq);
        for i in range(l):
            temp=[];
            for j in range(25):
                bin_prof = binary_profile_1(seq, j);
                if(j>=25):
                    print('Error! Feature Number must be between 0-24');
                    break;
                k=0;
                num=0;
                denom=0;
                ones=0;
                zeros=0;
                for j in range(len(bin_prof[i])):
                    if(bin_prof[i][j]==0):
                        num+=k*k;
                        denom+=k;
                        k=0;
                        zeros+=1;
                    elif(j==len(bin_prof[i])-1):
                        k+=1;
                        num+=k*k;
                        denom+=k;
                    else:
                        k+=1;
                        ones+=1;
                if(ones!=0):
                    answer = num/(ones*ones)
                    temp.append(round(num/(ones*ones),2));
                elif(ones==0):
                    temp.append(0);
            dist.append(temp)
        out = pd.DataFrame(dist)
        file1 = open(out123,'w')
        with file1:
            writer = csv.writer(file1);
            writer.writerows(dist);
        return out

    def lookup(peptide,featureNum):
        l=len(peptide);
        peptide = list(peptide);
        out=np.zeros(l);
        peptide_num = encode(peptide);
        for i in range(l):
            out[i] = PCP[peptide_num[i]][featureNum];
        return sum(out);
    def pcp(file):
        SEP_headers = ['SEP_PC','SEP_NC','SEP_NE','SEP_PO','SEP_NP','SEP_AL','SEP_CY','SEP_AR','SEP_AC','SEP_BS','SEP_NE_pH','SEP_HB','SEP_HL','SEP_NT','SEP_HX','SEP_SC','SEP_SS_HE','SEP_SS_ST','SEP_SS_CO','SEP_SA_BU','SEP_SA_EX','SEP_SA_IN','SEP_TN','SEP_SM','SEP_LR']
        if(type(file) == str):
            seq = pd.read_csv(file,header=None, sep=',');
            seq=seq.T
            seq[0].values.tolist()
            seq=seq[0];
        else:
            seq  = file;
        l = len(seq);
        rows = PCP.shape[0]; # Number of features in our reference table
        col = 20 ; # Denotes the 20 amino acids
        seq=[seq[i].upper() for i in range(l)]
        sequenceFeature = [];
        sequenceFeature.append(SEP_headers); #To put property name in output csv

        for i in range(l): # Loop to iterate over each sequence
            nfeatures = rows;
            sequenceFeatureTemp = [];
            for j in range(nfeatures): #Loop to iterate over each feature
                featureVal = lookup(seq[i],j)   
                if(len(seq[i])!=0):
                    sequenceFeatureTemp.append(featureVal/len(seq[i]))
                else:
                    sequenceFeatureTemp.append('NaN')
            sequenceFeature.append(sequenceFeatureTemp);
        out = pd.DataFrame(sequenceFeature);
        return sequenceFeature;
    def phyChem(file,mode='all',m=0,n=0):
        if(type(file) == str):
            seq1 = pd.read_csv(file,header=None, sep=',');
            seq1 = pd.DataFrame(seq1[0].str.upper())
            seq=[]
            [seq.append(seq1.iloc[i][0]) for i in range(len(seq1))]
        else:
            seq  = file;
        l = len(seq);
        newseq = [""]*l; # To store the n-terminal sequence
        for i in range(0,l):
            l = len(seq[i]);
            if(mode=='NT'):
                n=m;
                if(n!=0):
                    newseq[i] = seq[i][0:n];
                elif(n>l):
                    print('Warning! Sequence',i,"'s size is less than n. The output table would have NaN for this sequence");
                else:
                    print('Value of n is mandatory, it cannot be 0')
                    break;
            elif(mode=='CT'):
                n=m;
                if(n!=0):
                    newseq[i] = seq[i][(len(seq[i])-n):]
                elif(n>l):
                    print('WARNING: Sequence',i+1,"'s size is less than the value of n given. The output table would have NaN for this sequence");
                else:
                    print('Value of n is mandatory, it cannot be 0')
                    break;
            elif(mode=='all'):
                newseq = seq;
            elif(mode=='rest'):
                if(m==0):
                    print('Kindly provide start index for rest, it cannot be 0');
                    break;
                else:
                    if(n<=len(seq[i])):
                        newseq[i] = seq[i][m-1:n+1]
                    elif(n>len(seq[i])):
                        newseq[i] = seq[i][m-1:len(seq[i])]
                        print('WARNING: Since input value of n for sequence',i+1,'is greater than length of the protein, entire sequence starting from m has been considered')
            else:
                print("Wrong Mode. Enter 'NT', 'CT','all' or 'rest'");        
        output = pcp(newseq);
        return output
    def shannons(filename,out123):
        SEP_headers = ['SEP_PC','SEP_NC','SEP_NE','SEP_PO','SEP_NP','SEP_AL','SEP_CY','SEP_AR','SEP_AC','SEP_BS','SEP_NE_pH','SEP_HB','SEP_HL','SEP_NT','SEP_HX','SEP_SC','SEP_SS_HE','SEP_SS_ST','SEP_SS_CO','SEP_SA_BU','SEP_SA_EX','SEP_SA_IN','SEP_TN','SEP_SM','SEP_LR']
        if(type(filename) == str):
            seq1 = pd.read_csv(filename,header=None, sep=',');
            seq1 = pd.DataFrame(seq1[0].str.upper())
        else:
            seq1  = filename;
        seq=[]
        [seq.append(seq1.iloc[i][0]) for i in range(len(seq1))]
        comp = phyChem(seq);
        new = [comp[i][0:25] for i in range(len(comp))]
        entropy  = [];
        entropy.append(SEP_headers[0:25])
        for i in range(1,len(new)):
            seqEntropy = [];
            for j in range(len(new[i])):
                p = new[i][j]; 
                if((1-p) == 0. or p==0.):
                    temp = 0;#to store entropy of each sequence
                else:
                    temp = -(p*math.log2(p)+(1-p)*math.log2(1-p));
                seqEntropy.append(round(temp,3));
            entropy.append(seqEntropy);
        out = pd.DataFrame(entropy);
        file = open(out123,'w')
        with file:
            writer = csv.writer(file);
            writer.writerows(entropy);
        return entropy;

    ##################paac####################
    def val(AA_1, AA_2, aa, mat):
        return sum([(mat[i][aa[AA_1]] - mat[i][aa[AA_2]]) ** 2 for i in range(len(mat))]) / len(mat)
    def paac_1(file,lambdaval,w=0.05):
        data1 = pd.read_csv(nf_path+"/Data/data", sep = "\t")
        filename, file_extension = os.path.splitext(file)
        df = pd.read_csv(file, header = None)
        df1 = pd.DataFrame(df[0].str.upper())
        dd = []
        cc = []
        pseudo = []
        aa = {}
        for i in range(len(std)):
            aa[std[i]] = i
        for i in range(0,3):
            mean = sum(data1.iloc[i][1:])/20
            rr = math.sqrt(sum([(p-mean)**2 for p in data1.iloc[i][1:]])/20)
            dd.append([(p-mean)/rr for p in data1.iloc[i][1:]])
            zz = pd.DataFrame(dd)
        head = []
        for n in range(1, lambdaval + 1):
            head.append('_lam' + str(n))
        head = ['PAAC'+str(lambdaval)+sam for sam in head]
        pp = pd.DataFrame()
        ee = []
        for k in range(0,len(df1)):
            cc = []
            pseudo1 = [] 
            for n in range(1,lambdaval+1):
                cc.append(sum([val(df1[0][k][p], df1[0][k][p + n], aa, dd) for p in range(len(df1[0][k]) - n)]) / (len(df1[0][k]) - n))
                qq = pd.DataFrame(cc)
            pseudo = pseudo1 + [(w * p) / (1 + w * sum(cc)) for p in cc]
            ee.append(pseudo)
            ii = round(pd.DataFrame(ee, columns = head),4)
        ii.to_csv(filename+".lam",index = None)

    def paac(file,lambdaval,out,w=0.05):
        filename, file_extension = os.path.splitext(file)
        paac_1(file,lambdaval,w=0.05)
        aac_comp(file,filename+".aac")
        data1 = pd.read_csv(filename+".aac")
        header = ['PAAC'+str(lambdaval)+'_A','PAAC'+str(lambdaval)+'_C','PAAC'+str(lambdaval)+'_D','PAAC'+str(lambdaval)+'_E','PAAC'+str(lambdaval)+'_F','PAAC'+str(lambdaval)+'_G','PAAC'+str(lambdaval)+'_H','PAAC'+str(lambdaval)+'_I','PAAC'+str(lambdaval)+'_K','PAAC'+str(lambdaval)+'_L','PAAC'+str(lambdaval)+'_M','PAAC'+str(lambdaval)+'_N','PAAC'+str(lambdaval)+'_P','PAAC'+str(lambdaval)+'_Q','PAAC'+str(lambdaval)+'_R','PAAC'+str(lambdaval)+'_S','PAAC'+str(lambdaval)+'_T','PAAC'+str(lambdaval)+'_V','PAAC'+str(lambdaval)+'_W','PAAC'+str(lambdaval)+'_Y','Un']	
        data1.columns = header    
        data2 = pd.read_csv(filename+".lam")
        data3 = pd.concat([data1.iloc[:,:-1],data2], axis = 1).reset_index(drop=True)
        data3.to_csv(out,index=None)
        os.remove(filename+".lam")
        os.remove(filename+".aac")
    ######################apaac############################	
    def apaac_1(file,lambdaval,w=0.05):
        data1 = pd.read_csv(nf_path+"/Data/data", sep = "\t")
        filename, file_extension = os.path.splitext(file)
        df = pd.read_csv(file, header = None)
        df1 = pd.DataFrame(df[0].str.upper())
        dd = []
        cc = []
        pseudo = []
        aa = {}
        for i in range(len(std)):
            aa[std[i]] = i
        for i in range(0,3):
            mean = sum(data1.iloc[i][1:])/20
            rr = math.sqrt(sum([(p-mean)**2 for p in data1.iloc[i][1:]])/20)
            dd.append([(p-mean)/rr for p in data1.iloc[i][1:]])
            zz = pd.DataFrame(dd)
        head = []
        for n in range(1, lambdaval + 1):
            for e in ('HB','HL','SC'):
                head.append(e+'_lam' + str(n))
        head = ['APAAC'+str(lambdaval)+'_'+sam for sam in head]
        pp = pd.DataFrame()
        ee = []
        for k in range(0,len(df1)):
            cc = [] 
            for n in range(1,lambdaval+1):
                for b in range(0,len(zz)):
                    cc.append(sum([zz.loc[b][aa[df1[0][k][p]]] * zz.loc[b][aa[df1[0][k][p + n]]] for p in range(len(df1[0][k]) - n)]) / (len(df1[0][k]) - n))
                    qq = pd.DataFrame(cc)
            pseudo = [(w * p) / (1 + w * sum(cc)) for p in cc]
            ee.append(pseudo)
            ii = round(pd.DataFrame(ee, columns = head),4)
        ii.to_csv(filename+".plam",index = None)

    def apaac(file,lambdaval,out,w=0.05):
        filename, file_extension = os.path.splitext(file)
        apaac_1(file,lambdaval,w=0.05)
        aac_comp(file,filename+".aac")
        data1 = pd.read_csv(filename+".aac")
        headaac = []
        for i in std:
            headaac.append('APAAC'+str(lambdaval)+'_'+i)
        headaac.insert(len(headaac),0)
        data1.columns = headaac
        data2 = pd.read_csv(filename+".plam")
        data3 = pd.concat([data1.iloc[:,:-1],data2], axis = 1).reset_index(drop=True)
        data3.to_csv(out, index = None)
        os.remove(filename+".plam")
        os.remove(filename+".aac")
    ###################################qos#######################################
    def qos(file,gap,out,w=0.1):
        ff = []
        filename, file_extension = os.path.splitext(file)
        df = pd.read_csv(file, header = None)
        df2 = pd.DataFrame(df[0].str.upper())
        for i in range(0,len(df2)):
            ff.append(len(df2[0][i]))
        if min(ff) < gap:
            print("Error: All sequences' length should be higher than :", gap)
        else:
            mat1 = pd.read_csv(nf_path+"/Data/Schneider-Wrede.csv", index_col = 'Name')
            mat2 = pd.read_csv(nf_path+"/Data/Grantham.csv", index_col = 'Name')
            s1 = []
            s2 = []
            for i in range(0,len(df2)):
                for n in range(1, gap+1):
                    sum1 = 0
                    sum2 = 0
                    for j in range(0,(len(df2[0][i])-n)):
                        sum1 = sum1 + (mat1[df2[0][i][j]][df2[0][i][j+n]])**2
                        sum2 = sum2 + (mat2[df2[0][i][j]][df2[0][i][j+n]])**2
                    s1.append(sum1)
                    s2.append(sum2)
            zz = pd.DataFrame(np.array(s1).reshape(len(df2),gap))
            zz["sum"] = zz.sum(axis=1)
            zz2 = pd.DataFrame(np.array(s2).reshape(len(df2),gap))
            zz2["sum"] = zz2.sum(axis=1)
            c1 = []
            c2 = []
            c3 = []
            c4 = []
            h1 = []
            h2 = []
            h3 = []
            h4 = []
            for aa in std:
                h1.append('QSO'+str(gap)+'_SC_' + aa)
            for aa in std:
                h2.append('QSO'+str(gap)+'_G_' + aa)
            for n in range(1, gap+1):
                h3.append('SC' + str(n))
            h3 = ['QSO'+str(gap)+'_'+sam for sam in h3]
            for n in range(1, gap+1):
                h4.append('G' + str(n))
            h4 = ['QSO'+str(gap)+'_'+sam for sam in h4]
            for i in range(0,len(df2)):
                AA = {}
                for j in std:
                    AA[j] = df2[0][i].count(j)
                    c1.append(AA[j] / (1 + w * zz['sum'][i]))
                    c2.append(AA[j] / (1 + w * zz2['sum'][i]))
                for k in range(0,gap):
                    c3.append((w * zz[k][i]) / (1 + w * zz['sum'][i]))
                    c4.append((w * zz[k][i]) / (1 + w * zz['sum'][i]))
            pp1 = np.array(c1).reshape(len(df2),len(std))
            pp2 = np.array(c2).reshape(len(df2),len(std))
            pp3 = np.array(c3).reshape(len(df2),gap)
            pp4 = np.array(c4).reshape(len(df2),gap)
            zz5 = round(pd.concat([pd.DataFrame(pp1, columns = h1),pd.DataFrame(pp2,columns = h2),pd.DataFrame(pp3, columns = h3),pd.DataFrame(pp4, columns = h4)], axis = 1),4)
            zz5.to_csv(out, index = None, encoding = 'utf-8')	
    ##########################soc################
    def soc(file,gap,out):
        ff = []
        filename, file_extension = os.path.splitext(file)
        df = pd.read_csv(file, header = None)
        df2 = pd.DataFrame(df[0].str.upper())
        for i in range(0,len(df2)):
            ff.append(len(df2[0][i]))
        if min(ff) < gap:
            print("Error: All sequences' length should be higher than :", gap)
            return 0
        mat1 = pd.read_csv(nf_path+"/Data/Schneider-Wrede.csv", index_col = 'Name')
        mat2 = pd.read_csv(nf_path+"/Data/Grantham.csv", index_col = 'Name')
        h1 = []
        h2 = []
        for n in range(1, gap+1):
            h1.append('SC' + str(n))
        for n in range(1, gap + 1):
            h2.append('G' + str(n))
        h1 = ['SOC'+str(gap)+'_'+sam for sam in h1]
        h2 = ['SOC'+str(gap)+'_'+sam for sam in h2]
        s1 = []
        s2 = []
        for i in range(0,len(df2)):
            for n in range(1, gap+1):
                sum = 0
                sum1 =0
                sum2 =0
                sum3 =0
                for j in range(0,(len(df2[0][i])-n)):
                    sum = sum + (mat1[df2[0][i][j]][df2[0][i][j+n]])**2
                    sum1 = sum/(len(df2[0][i])-n)
                    sum2 = sum2 + (mat2[df2[0][i][j]][df2[0][i][j+n]])**2
                    sum3 = sum2/(len(df2[0][i])-n)
                s1.append(sum1)
                s2.append(sum3)
        zz = np.array(s1).reshape(len(df2),gap)
        zz2 = np.array(s2).reshape(len(df2),gap)
        zz3 = round(pd.concat([pd.DataFrame(zz, columns = h1),pd.DataFrame(zz2,columns = h2)], axis = 1),4)
        zz3.to_csv(out, index = None, encoding = 'utf-8') 
    ##########################################CTC###################################
    import pandas as pd
    import csv

    def repstring(string):
        string = string.upper()
        char = {"A":"1", "G":"1", "V":"1", "I":"2", "L":"2", "F":"2", "P":"2", "Y":"3", "M":"3", "T":"3", "S":"3", "H":"4", "N":"4", "Q":"4", "W":"4", "R":"5", "K":"5", "D":"6", "E":"6", "C":"7"}
        string = list(string)
        for index, item in enumerate(string):
            for key, value in char.items():
                if item == key:
                    string[index] = value
        return "".join(string)

    def occurrences(string, sub_string):
        count = 0
        beg = 0
        while(string.find(sub_string, beg) != -1):
            count += 1
            beg = string.find(sub_string, beg)
            beg += 1
        return count

    def CTC(filename, out, triads):
        data = list((pd.read_csv(filename, sep=',', header=None)).iloc[:, 0])
        result = []

        for sequence in data:
            sequence = sequence.upper()
            Y = repstring(sequence)
            row = []
            for triad in triads:
                row.append(round(occurrences(Y, triad) / len(Y), 3))
            result.append(row)

        headers = ['CTC_' + triad for triad in triads]

        with open(out, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(result)

    # List of triads for which you want to calculate features
    triads_to_calculate = [
        '642', '464', '735', '517', '326', '246', '552', '452', '127', '273',
        '547', '557', '251', '725', '472', '115', '274', '574', '665', '666',
        '333', '525', '575', '512', '155', '471', '644', '626', '211', '255',
        '225', '566', '227', '745', '715', '677', '522', '713', '112', '122',
        '174', '221', '173', '545', '661', '355'
    ]


    ######################################CETD###################################
    def ctd(file,out):
        attr=pd.read_csv(nf_path+"/Data/aa_attr_group.csv", sep="\t")
        filename, file_extension = os.path.splitext(file)
        df1 = pd.read_csv(file, header = None)
        df = pd.DataFrame(df1[0].str.upper())
        n = 0
        stt1 = []
        m = 1
        for i in range(0,len(attr)) :
            st =[]
            stt1.append([])
            for j in range(0,len(df)) :
                stt1[i].append([])
                for k in range(0,len(df[0][j])) :
                    while m < 4 :
                        while n < len(attr.iloc[i,m]) :
                            if df[0][j][k] == attr.iloc[i,m][n] :
                                st.append(m)
                                stt1[i][j].append(m)
                            n += 2
                        n = 0
                        m += 1
                    m = 1
    #####################Composition######################
        f = open("compout_1", 'w')
        sys.stdout = f
        std = [1,2,3]
        print("1,2,3,")
        for p in range (0,len(df)) :
            for ii in range(0,len(stt1)) :
                #for jj in stt1[ii][p]:
                for pp in std :
                    count = 0
                    for kk in stt1[ii][p] :
                        temp1 = kk
                        if temp1 == pp :
                            count += 1
                        composition = (count/len(stt1[ii][p]))*100
                    print("%.2f"%composition, end = ",")
                print("")
        f.truncate()

    #################################Transition#############
        tt = []
        tr=[]
        kk =0
        for ii in range(0,len(stt1)) :
            tt = []
            tr.append([])
            for p in range (0,len(df)) :
                tr[ii].append([])
                while kk < len(stt1[ii][p]) :
                    if kk+1 <len(stt1[ii][p]):
                    #if  stt1[ii][p][kk] < stt1[ii][p][kk+1] or stt1[ii][p][kk] > stt1[ii][p][kk+1]: # condition for adjacent values
                        tt.append(stt1[ii][p][kk])
                        tt.append(stt1[ii][p][kk+1])
                        tr[ii][p].append(stt1[ii][p][kk])
                        tr[ii][p].append(stt1[ii][p][kk+1])

                    kk += 1
                kk = 0

        pp = 0
        xx = []
        xxx = []
        for mm in range(0,len(tr)) :
            xx = []
            xxx.append([])
            for nn in range(0,len(tr[mm])):
                xxx[mm].append([])
                while pp < len(tr[mm][nn]) :
                    xx .append(tr[mm][nn][pp:pp+2])
                    xxx[mm][nn].append(tr[mm][nn][pp:pp+2])
                    pp+=2
                pp = 0

        f1 = open("compout_2", 'w')
        sys.stdout = f1
        std1 = [[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]]
        print("1->1,1->2,1->3,2->1,2->2,2->3,3->1,3->2,3->3,")
        for rr in range(0,len(df)) :
            for qq in range(0,len(xxx)):
                for tt in std1 :
                    count = 0
                    for ss in xxx[qq][rr] :
                        temp2 = ss
                        if temp2 == tt :
                            count += 1
                    print(count, end = ",")
                print("")
        f1.truncate()

        #################################Distribution#############
        c_11 = []
        c_22 = []
        c_33 = []
        zz = []
        #print("0% 25% 50% 75% 100%")
        for x in range(0,len(stt1)) :
            #c_11.append([])
            c_22.append([])
            #c_33.append([])
            yy_c_1 = []
            yy_c_2 = []
            yy_c_3 = []
            ccc = []

            k = 0
            j = 0
            for y in range(0,len(stt1[x])):
                #c_11[x].append([])
                c_22[x].append([])
                for i in range(1,4) :
                    cc = []
                    c1 = [index for index,value in enumerate(stt1[x][y]) if value == i]
                    c_22[x][y].append(c1)
        cc = []
        for ss in range(0,len(df)):
            for uu in range(0,len(c_22)):
                for mm in range(0,3):
                    for ee in range(0,101,25):
                        k = (ee*(len(c_22[uu][ss][mm])))/100
                        cc.append(math.floor(k))
        f2 = open('compout_3', 'w')
        sys.stdout = f2
        print("0% 25% 50% 75% 100%")
        for i in range (0,len(cc),5):
            print(*cc[i:i+5])
        f2.truncate()
        head = []
        header1 = ['CeTD_HB','CeTD_VW','CeTD_PO','CeTD_PZ','CeTD_CH','CeTD_SS','CeTD_SA']
        for i in header1:
            for j in range(1,4):
                head.append(i+str(j))
        df11 = pd.read_csv("compout_1")
        df_1 = df11.iloc[:,:-1]
        zz = pd.DataFrame()
        for i in range(0,len(df_1),7):
            zz = pd.concat([zz, pd.DataFrame(pd.concat([df_1.loc[i],df_1.loc[i+1],df_1.loc[i+2],df_1.loc[i+3],df_1.loc[i+4],df_1.loc[i+5],df_1.loc[i+6]],axis=0)).transpose()], ignore_index=True)
        zz.columns = head
        #zz.to_csv(filename+".ctd_comp", index=None, encoding='utf-8')
        head2 = []
        header2 = ['CeTD_11','CeTD_12','CeTD_13','CeTD_21','CeTD_22','CeTD_23','CeTD_31','CeTD_32','CeTD_33']
        for i in header2:
            for j in ('HB','VW','PO','PZ','CH','SS','SA'):
                head2.append(i+'_'+str(j))
        df12 = pd.read_csv("compout_2")
        df_2 = df12.iloc[:,:-1]
        ss = pd.DataFrame()
        for i in range(0,len(df_2),7):
            ss = pd.concat([ss, pd.DataFrame(pd.concat([df_2.loc[i], df_2.loc[i+1], df_2.loc[i+2], df_2.loc[i+3], df_2.loc[i+4], df_2.loc[i+5], df_2.loc[i+6]], axis=0)).transpose()], ignore_index=True)
        ss.columns = head2
        head3 = []
        header3 = ['CeTD_0_p','CeTD_25_p','CeTD_50_p','CeTD_75_p','CeTD_100_p']
        header4 = ['HB','VW','PO','PZ','CH','SS','SA']
        for j in range(1,4):
            for k in header4:
                for i in header3:
                    head3.append(i+'_'+k+str(j))
        df_3 = pd.read_csv("compout_3", sep=" ")
        rr = pd.DataFrame()
        for i in range(0,len(df_3),21):
            rr = pd.concat([rr, pd.DataFrame(pd.concat([df_3.loc[i], df_3.loc[i+1], df_3.loc[i+2], df_3.loc[i+3], df_3.loc[i+4], df_3.loc[i+5], df_3.loc[i+6], df_3.loc[i+7], df_3.loc[i+8], df_3.loc[i+9], df_3.loc[i+10], df_3.loc[i+11], df_3.loc[i+12], df_3.loc[i+13], df_3.loc[i+14], df_3.loc[i+15], df_3.loc[i+16], df_3.loc[i+17], df_3.loc[i+18], df_3.loc[i+19], df_3.loc[i+20]], axis=0)).transpose()], ignore_index=True)
        rr.columns = head3
        cotrdi= pd.concat([zz,ss,rr],axis=1)
        cotrdi.to_csv(out, index=None, encoding='utf-8')
        os.remove('compout_1')
        os.remove('compout_2')
        os.remove('compout_3')

    # Remove columns starting with "Unnamed:" after calling each function
    aac_out_file = ("aac_output2")
    aac_comp(file_path, aac_out_file)

    dpc_out_file = ("dpc_output2")
    dpc_comp(file_path,1, dpc_out_file,selected_dipeptides)

    tpc_out_file = ("tpc_output2")
    tpc_comp(file_path, tpc_out_file,selected_tripeptides)

    atc_out_file = ("atc_output2")
    atc(file_path, atc_out_file)

    btc_out_file = ("btc_output2")
    bond(file_path, btc_out_file)

    pcp_1_out_file = ("pcp_output2")
    pcp_1(file_path, pcp_1_out_file)

    rri_res_out_file = ("rri_output2")
    RAAC(file_path, rri_res_out_file)

    pri_res_out_file = ("pri_output2")
    repeats(file_path, pri_res_out_file)

    ddor_out_file = ("ddor_output2")
    DDOR(file_path, ddor_out_file)

    sep_out_file = ("sep_output2")
    SE(file_path, sep_out_file)

    ser_res_out_file = ("ser_output2")
    SE_residue_level(file_path, ser_res_out_file)

    spc_res_out_file = ("spc_output2")
    shannons(file_path, spc_res_out_file)

    paac_res_out_file = ("paac_output2")
    paac(file_path, 1, paac_res_out_file)

    apaac_out_file = ("apaac_output2")
    apaac(file_path, 1, apaac_out_file)

    qso_out_file = ("qso_output2")
    qos(file_path, 1, qso_out_file)

    soc_out_file = ("soc_output2")
    soc(file_path, 1, soc_out_file)

    ctc_out_file = ("ctc_output2")
    CTC(file_path, ctc_out_file,triads_to_calculate)

    cetd_out_file = ("cetd_output2")
    ctd(file_path, cetd_out_file)

    dfs = []
    for out_file in [aac_out_file, dpc_out_file, tpc_out_file, atc_out_file, btc_out_file, pcp_1_out_file,
                     rri_res_out_file, pri_res_out_file, ddor_out_file, sep_out_file, ser_res_out_file,
                     spc_res_out_file, paac_res_out_file, apaac_out_file, qso_out_file,soc_out_file, ctc_out_file, cetd_out_file]:
        df = pd.read_csv(out_file)
        df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]
        dfs.append(df)
    concatenated_df = pd.concat(dfs, axis=1)
    concatenated_df.to_csv(output_file, index=False)
    os.remove('aac_output2')
    os.remove('dpc_output2')
    os.remove('tpc_output2')
    os.remove('atc_output2')
    os.remove('btc_output2')
    os.remove('pcp_output2')
    os.remove('rri_output2')
    os.remove('pri_output2')
    os.remove('ddor_output2')
    os.remove('sep_output2')
    os.remove('ser_output2')
    os.remove('spc_output2')
    os.remove('paac_output2')
    os.remove('apaac_output2')
    os.remove('qso_output2')
    os.remove('soc_output2')
    os.remove('ctc_output2')
    os.remove('cetd_output2')


# Function to check the seqeunce
def readseq(file):
    with open(file) as f:
        records = f.read()
    records = records.split('>')[1:]
    seqid = []
    seq = []
    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '', ''.join(array[1:]).upper())
        seqid.append('>'+name)
        seq.append(sequence)
    if len(seqid) == 0:
        f=open(file,"r")
        data1 = f.readlines()
        for each in data1:
            seq.append(each.replace('\n',''))
        for i in range (1,len(seq)+1):
            seqid.append(">Seq_"+str(i))
    df1 = pd.DataFrame(seqid)
    df1.columns =['ID']
    df2 = pd.DataFrame(seq)
    df2.columns =['Seq']
    final_df = pd.concat([pd.DataFrame(seqid),pd.DataFrame(seq)], axis=1)
    return df1,df2

# Function to check the length of seqeunces
def lenchk(file1):
    cc = []
    df1 = file1
    df1.columns = ['seq']
    for i in range(len(df1)):
        if len(df1['seq'][i])>30:
            cc.append(df1['seq'][i][0:30])
        else:
            cc.append(df1['seq'][i])
    df2 = pd.DataFrame(cc)
    df2.to_csv('out_len', index = None , header = None)
    df2.columns = ['Seq']
    return df2

def seq_pattern(file1,file2,num):
    df1 = file1
    df1.columns = ['Seq']
    df2 = file2
    df2.columns = ['Name']
    cc = []
    dd = []
    ee = []
    for i in range(len(df1)):
        for j in range(len(df1['Seq'][i])):
            xx = df1['Seq'][i][j:j + num]
            if len(xx) == num:
                cc.append(df2['Name'][i])
#                 dd.append('Pattern_'+str(j+1)+'_Seq'+str(i+1))
                dd.append(f'Pattern_{j + 1}_{df2["Name"][i]}')
                ee.append(xx)
    df3 = pd.concat([pd.DataFrame(cc),pd.DataFrame(dd),pd.DataFrame(ee)],axis=1)
    df3.columns= ['Seq_ID','Pattern_ID','Seq']
    return df3

def readseq_1(file):
    with open(file) as f:
        records = f.read()
    records = records.split('>')[1:]
    seqid = []
    seq = []
    
    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '', ''.join(array[1:]).upper())
        seqid.append('>' + name)
        seq.append(sequence)
    
    if len(seqid) == 0:
        f = open(file, "r")
        data1 = f.readlines()
        for each in data1:
            seq.append(each.replace('\n', ''))
        for i in range(1, len(seq) + 1):
            seqid.append(">Seq_" + str(i))
    
    df1 = pd.DataFrame(seqid, columns=['ID'])
    df2 = pd.DataFrame(seq, columns=['Seq'])
    final_df = pd.concat([df1, df2], axis=1)

    path ='./file_in/'

    for index, row in final_df.iterrows():
        file_name = row['ID'].replace('>', '') + '.fasta'
        file_path = os.path.join(path, file_name)
        row.to_frame().T.to_csv(file_path, index=False, header=False, sep='\n')
    
    return final_df


def ml_scan(file,filename):
    df_p = pd.read_csv(file)
    df_p.columns = range(df_p.shape[1])
    df1_p = df_p.iloc[:, 2]
    df1_p.to_csv('out_len', index = None, header = None)
    features_calulate('out_len', 'out1')
    a=[]
    df = pd.read_csv('out1')
    ff = pd.read_csv(nf_path+'/Data/selected_features_mrmr1000_new.csv')
    ff2 = ff["SelectedFeatures"].tolist()
    aa = pd.concat([df[ff2]], axis =1)
    clf = pickle.load(open(nf_path+'/Data/model2.pkl','rb'))
    data_test = aa
    X_test = data_test
    y_p_score1=clf.predict(X_test)
    y_p_s1=y_p_score1.tolist()
    a.extend(y_p_s1)
    df = pd.DataFrame(a)
    df1 = df.iloc[:,-1].round(3)
    df2 = pd.DataFrame(df1)
    df2.columns = ['MIC']
    # dd = pd.concat([aa,df2], axis =1)
    df3 = pd.concat([df_p,df2],axis=1)
    df3['MIC(uM)'] = 2 ** df3['MIC']
    df4 = df3.drop(['MIC'], axis =1)
    df4.columns  = ['ID', 'Pattern', 'Sequence', 'MIC(uM)']
    df4.to_csv(filename, index=None)
    os.remove('out_len')
    os.remove('out1')
    return df4 


def scan_2(file, out, Win_len):
    os.makedirs('file_out', exist_ok = True)
    os.makedirs('file_in', exist_ok = True)
    readseq_1(file)

    folder_path = './file_in/'  
    output_folder_path = './file_out/' 
    os.makedirs(output_folder_path, exist_ok=True)
    fasta_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.fasta')])

    pattern_length = int(Win_len) 

    all_results = []  

    # Process each FASTA file one by one in sorted order
    for fasta_file in tqdm(fasta_files, desc="Processing FASTA files"):
        fasta_file_path = os.path.join(folder_path, fasta_file)  
        print(f"Processing file: {fasta_file_path}") 
        df_2, dfseq = readseq(fasta_file_path)
        patterns_df = seq_pattern(dfseq, df_2, pattern_length)

        output_file_name = os.path.splitext(fasta_file)[0] + '_patterns.csv'  
        pattern_output_file_path = os.path.join(output_folder_path, output_file_name) 
        patterns_df.to_csv(pattern_output_file_path, index=False)

        print(f"Patterns saved to: {pattern_output_file_path}")  

        mic_results_output_file_name = os.path.splitext(fasta_file)[0] + '_results.csv' 
        mic_output_file_path = os.path.join(output_folder_path, mic_results_output_file_name)  

        mic_results_df = ml_scan(pattern_output_file_path, mic_output_file_path)  
        if os.path.exists(pattern_output_file_path):
            os.remove(pattern_output_file_path)
            print(f"Deleted: {pattern_output_file_path}")

        all_results.append(mic_results_df)
    final_results_df = pd.concat(all_results, ignore_index=True)
    final_output_file_path = os.path.join(output_folder_path, out)
    final_results_df.to_csv(final_output_file_path, index=False)

    print(f"Final results saved to: {final_output_file_path}")
    for fasta_file in tqdm(fasta_files, desc="Deleting intermediate files"):
        mic_results_output_file_name = os.path.splitext(fasta_file)[0] + '_results.csv' 
        mic_output_file_path = os.path.join(output_folder_path, mic_results_output_file_name)  
        if os.path.exists(mic_output_file_path):
            os.remove(mic_output_file_path)
            print(f"Deleted: {mic_output_file_path}")

    import shutil
    shutil.rmtree('file_in')
    print("All intermediate files deleted.")

def mutants_old(file1,file2):
    std = list("ACDEFGHIKLMNPQRSTVWY")
    cc = []
    dd = []
    ee = []
    df2 = file2
    df2.columns = ['Name']
    df1 = file1
    df1.columns = ['Seq']
    for k in range(len(df1)):
        cc.append(df1['Seq'][k])
        dd.append('Original_'+'Seq'+str(k+1))
        ee.append(df2['Name'][k])
        for i in range(0,len(df1['Seq'][k])):
            for j in std:
                if df1['Seq'][k][i]!=j:
                    dd.append('Mutant_'+df1['Seq'][k][i]+str(i+1)+j+'_Seq'+str(k+1))
                    cc.append(df1['Seq'][k][:i] + j + df1['Seq'][k][i + 1:])
                    ee.append(df2['Name'][k])
    xx = pd.concat([pd.DataFrame(ee),pd.DataFrame(dd),pd.DataFrame(cc)],axis=1)
    xx.columns = ['Seq_ID','Mutant_ID','Seq']
    return xx
def ML_run(file1, out):
    a=[]
    df = pd.read_csv(file1)
    ff = pd.read_csv(nf_path+'/Data/selected_features_mrmr1000_new.csv')
    ff2 = ff["SelectedFeatures"].tolist()
    aa = pd.concat([df[ff2]], axis =1)
    clf = pickle.load(open(nf_path+'/Data/model2.pkl','rb'))
    data_test = aa
    X_test = data_test
    y_p_score1=clf.predict(X_test)
    y_p_s1=y_p_score1.tolist()
    a.extend(y_p_s1)
    df = pd.DataFrame(a)
    df1 = df.iloc[:,-1].round(3)
    df2 = pd.DataFrame(df1)
    df2.columns = ['MIC']
    dd = pd.concat([aa,df2], axis =1)
    dd.to_csv(out, index = None)
    return df2
def main():
    print('############################################################################################')
    print('# This program EIPPred is developed for predicting, desigining and scanning MIC of peptides #')
    print('# mellitus causing  peptides, developed by Prof G. P. S. Raghava group.               #')
    print('# Please cite: EIPPred; available at https://webs.iiitd.edu.in/raghava/eippred/  #')
    print('############################################################################################')

    parser = argparse.ArgumentParser(description='Please provide following arguments')

    ## Read Arguments from command
    parser.add_argument("-i", "--input", type=str, required=True, help="Input: protein or peptide sequence(s) in FASTA format or single sequence per line in single letter code")
    parser.add_argument("-o", "--output",type=str, help="Output: File for saving results by default outfile.csv")
    parser.add_argument("-j", "--job",type=int, choices = [1,2,3], help="Job Type: 1:Predict, 2: Design, 3: Scan by default 1")
    parser.add_argument("-w","--winleng", type=int, choices =range(8,31), help="Window Length: 8 to 30 (scan mode only), by default 9")


    args = parser.parse_args()
    # Parameter initialization or assigning variable for command level arguments
    Sequence= args.input  
    if args.output == None:
        result_filename= "outfile.csv"
    else:
        result_filename = args.output
    if args.job == None:
            Job = int(1)
    else:
            Job = int(args.job)
    # Window Length 
    if args.winleng == None:
            Win_len = int(9)
    else:
            Win_len = int(args.winleng)

    random_number = random.randint(1, 1000)
    folder_name = f"eippred_{random_number}"
    input_file_name = Sequence
    destination_file_path = os.path.join(os.getcwd(), input_file_name)
    os.makedirs(folder_name, exist_ok=True)
    shutil.copy(input_file_name, folder_name)
    os.chdir(folder_name)
    print(f"Copied '{input_file_name}' to '{folder_name}'.")
    print(f"Current working directory changed to: {os.getcwd()}")
    warnings.filterwarnings("ignore")
    #======================= Prediction Module start from here =====================
    if Job == 1:
        print('\n======= Thanks for using Predict module of EIPPred. Your results will be stored in file :',result_filename,' =====\n')
        df_2,dfseq = readseq(Sequence)
        df1 = lenchk(dfseq)
        features_calulate('out_len','out2')
        mlres = ML_run('out2', 'out4')
        df3 = pd.concat([df_2,df1,mlres],axis=1)
        df3['MIC(uM)'] = 2 ** df3['MIC']
        df4 = df3.drop(['MIC'], axis =1)
        df4.to_csv(result_filename,index = None)
        os.remove('out_len')
        os.remove('out2')
        os.remove('out4')
        print("\n=========Process Completed. Have an awesome day ahead.=============\n")

    #===================== Design Model Start from Here ======================
    elif Job == 2:
        print('\n======= Thanks for using Design module of Eippred. Your results will be stored in file :',result_filename,' =====\n')
        print('==== Designing Peptides: Processing sequences please wait ...')
        df_2,dfseq = readseq(Sequence)
        df_1 = mutants_old(dfseq,df_2)
        dfseq = df_1[['Seq']]
        df1 = lenchk(dfseq)
        features_calulate('out_len','out2')
        mlres = ML_run('out2', 'out4')
        mlres = mlres.round(3)
        df_1['Mutant'] = ['>'+df_1['Mutant_ID'][i] for i in range(len(df_1))]
        df3 = pd.concat([df_1,mlres],axis=1)
        df3['MIC(uM)'] = 2 ** df3['MIC']
        df4 = df3.drop(['MIC'], axis =1)
        df4.to_csv(result_filename, index=None)
        os.remove('out_len')
        os.remove('out2')
        os.remove('out4')
        print("\n=========Process Completed. Have an awesome day ahead.=============\n")

    elif Job==3:
        print('\n======= Thanks for using Scan module of EIPPred. Your results will be stored in file :',result_filename,' =====\n')
        print('==== Scanning Peptides: Processing sequences please wait ...')
        scan_2(Sequence, result_filename, Win_len)
        print("\n=========Process Completed. Have an awesome day ahead.=============\n")

if __name__ == "__main__":
    main()


def calculate_basic_features_eipred(sequences, selected_features):
    """Calculate basic AAC, DPC, TPC features from sequences for EIPred."""
    std_aa = "ACDEFGHIKLMNPQRSTVWY"
    
    # Initialize feature matrix
    feature_matrix = pd.DataFrame(np.zeros((len(sequences), len(selected_features))), 
                                 columns=selected_features)
    
    for seq_idx, sequence in enumerate(sequences):
        seq = sequence.upper()
        seq_len = len(seq)
        
        # Calculate AAC (Amino Acid Composition)
        for aa in std_aa:
            aac_name = f"AAC_{aa}"
            if aac_name in selected_features:
                count = seq.count(aa)
                composition = (count / seq_len) * 100 if seq_len > 0 else 0
                feature_matrix.at[seq_idx, aac_name] = composition
        
        # Calculate DPC (Dipeptide Composition) 
        if seq_len > 1:
            for i in range(seq_len - 1):
                dipeptide = seq[i:i+2]
                if len(dipeptide) == 2 and all(aa in std_aa for aa in dipeptide):
                    dpc_name = f"DPC_{dipeptide}"
                    if dpc_name in selected_features:
                        feature_matrix.at[seq_idx, dpc_name] += (1 / (seq_len - 1)) * 100
        
        # Calculate TPC (Tripeptide Composition)
        if seq_len > 2:
            for i in range(seq_len - 2):
                tripeptide = seq[i:i+3]
                if len(tripeptide) == 3 and all(aa in std_aa for aa in tripeptide):
                    tpc_name = f"TPC_{tripeptide}"
                    if tpc_name in selected_features:
                        feature_matrix.at[seq_idx, tpc_name] += (1 / (seq_len - 2)) * 100
    
    return feature_matrix


class EIPredPredictor:
    """EIPred predictor with APEX-compatible interface.
    
    Interface matches PredictorAPEX:
    - __init__(device='cpu', batch_size=1000): constructor
    - predict(seq_list): returns MIC predictions in uM, shape (n_sequences, 1)
    """
    
    def __init__(
        self,
        device="cpu",
        batch_size=1000,
        model="default",
        models_directory=None,
    ):
        from pep_compass.optimization.components.oracles.model_registry import (
            resolve_oracle_model,
        )

        self.device = device  # Not used but kept for interface compatibility
        self.batch_size = batch_size
        self.nf_path = nf_path
        self.model_name = model
        _, model_paths = resolve_oracle_model(
            "eipred",
            model,
            models_directory=models_directory,
        )
        self.model_path, self.features_path = map(str, model_paths)
        self.clf = None
        self.selected_features = None
        
    def _load_model_and_features(self):
        """Lazy load model and feature list."""
        if self.clf is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"EIPred model not found: {self.model_path}")
            with open(self.model_path, 'rb') as f:
                self.clf = pickle.load(f)
                print("EIPred model loaded successfully.")
                
        if self.selected_features is None:
            if not os.path.exists(self.features_path):
                raise FileNotFoundError(f"EIPred features not found: {self.features_path}")
            ff = pd.read_csv(self.features_path)
            self.selected_features = ff['SelectedFeatures'].tolist()
            print("EIPred selected features loaded successfully.")
    
    def predict_log2(self, seq_list):
        """Predict log2(MIC) for a list of sequences (raw model output).
        
        Args:
            seq_list: list of strings (sequences)
            
        Returns:
            numpy array of shape (n_sequences, 1) with log2(MIC) values
        """
        self._load_model_and_features()
        
        # Calculate features from sequences
        X_test = calculate_basic_features_eipred(seq_list, self.selected_features)
        
        # Predict using model (returns log2 MIC values)
        y_pred = self.clf.predict(X_test)
        print(f"y_pred straight from the model: {y_pred}")
        
        # Return raw log2 MIC values
        return y_pred.reshape(-1, 1)
    
    def predict(self, seq_list):
        """Predict 10**(-y_pred) for a list of sequences.
        
        Args:
            seq_list: list of strings (sequences)
            
        Returns:
            numpy array of shape (n_sequences,) with 10**(-y_pred) values
        """
        # Get raw model predictions (y_pred)
        log2_preds = self.predict_log2(seq_list)
        y_pred = log2_preds.flatten()
        
        # Return 10**(-y_pred)
        return 10**(-y_pred)


class PredictorEIPred:
    """Lightweight wrapper exposing a .predict(seq_list) API.

    Supports two modes:
    - Full pipeline: if `Data` CSVs are present, runs the existing
      feature-calculation pipeline and uses the selected features.
    - Model-only fallback: if CSVs are missing, builds a zero-filled
      feature matrix sized to the classifier's expected input and
      performs inference. This is useful for quick integration tests.
    """

    def __init__(self, model_path=None):
        self.nf_path = nf_path
        if model_path is None:
            self.model_path = os.path.join(self.nf_path, 'Data', 'model2.pkl', 'model2.pkl')
        else:
            self.model_path = model_path
        self.clf = None

    def _load_clf(self):
        if self.clf is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"EIPred model not found: {self.model_path}")
            with open(self.model_path, 'rb') as f:
                self.clf = pickle.load(f)
        return self.clf

    def predict(self, seq_list):
        # create temporary working directory
        with tempfile.TemporaryDirectory() as td:
            out_len = os.path.join(td, 'out_len')
            out1 = os.path.join(td, 'out1')

            # write sequences
            with open(out_len, 'w') as f:
                for s in seq_list:
                    f.write(str(s).upper() + '\n')

            # decide whether to run full feature pipeline
            data_dir = os.path.join(self.nf_path, 'Data')
            physico_path = os.path.join(data_dir, 'PhysicoChemical.csv')
            sel_feat_path = os.path.join(data_dir, 'selected_features_mrmr1000_new.csv')

            if os.path.exists(physico_path) and os.path.exists(sel_feat_path):
                features_calulate(out_len, out1)
                df = pd.read_csv(out1)
                ff = pd.read_csv(sel_feat_path)
                ff2 = ff['SelectedFeatures'].tolist()
                X_test = pd.concat([df[ff2]], axis=1)
            else:
                # Fallback: calculate basic features directly from sequences
                ff = pd.read_csv(os.path.join(self.nf_path, 'DATA', 'selected_features_mrmr1000_new.csv'))
                selected_features = ff['SelectedFeatures'].tolist()
                X_test = calculate_basic_features_eipred(seq_list, selected_features)
                warnings.warn("EIPred feature CSVs missing; using basic AAC/DPC/TPC features.")

            clf = self._load_clf()
            y_p_score = clf.predict(X_test)
            try:
                mic_uM = (2 ** np.asarray(y_p_score)).reshape(-1, 1)
            except Exception:
                mic_uM = np.asarray(y_p_score).reshape(-1, 1)

            return mic_uM
