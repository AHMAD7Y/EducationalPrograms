// Coded By Line Style
// Accuracy ==    9801 / 10000    ==    98.01 %

#include"cmath"
#include"cstdio"
#include"ctime"

#define NumberOfTrainData 60000
#define NumberOfTestData 10000
#define Pixels 784 // 28*28
#define Layers 3
#define Neurons 1000

using namespace std;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Memory Performance:  static local memory   >   static global memory   >   dynamic local memory   >   dynamic global memory
// Memory Size:         static local memory   <   static global memory   <   dynamic local memory   <   dynamic global memory
// Addition Note:       any process on variables in local main() function --> ( faster than all , smaller than all )

// So i used fastest memory which size of it can fit of this problem --> static local memory ( not fit ) , static global memory ( fit )
// So i used static global memory for store all features of class MNIST_NN as you can see below.......

// Final Advice: don't use this approach unless project scope is small, when large projects use pure OOP --> use dynamic local memory --> put all these features in their class

char c[200000000];

unsigned int i,j,k,l, // k --> Data // l --> layers // i  --> left layer neuron // j right layer neuron
             Epoch,result,correct;

unsigned short neurons[Layers]={784,200, 10},
               TrainData[NumberOfTrainData][Pixels],TestData[NumberOfTestData][Pixels],
               TrainResult[NumberOfTrainData],TestResult[NumberOfTestData];

double  NeuronCombination[Layers][Neurons],
        NeuronActivation[Layers][Neurons],
        NeuronError[Layers][Neurons],
        NeuronWeight[Layers][Neurons][Neurons],
        NeuronDeltaWeight[Layers][Neurons][Neurons],
        SumWeights[Layers][Neurons],
        MomentumRate,SchedulingMomentumRate,LearningRate,SchedulingLearningRate,
        NewDelta,tmp;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class MNIST_NN
{
public:
    MNIST_NN()
    {
        Epoch=20;
        MomentumRate=0.7; SchedulingMomentumRate=1.0;
        LearningRate=0.3; SchedulingLearningRate=0.9;

        NeuronCombination[0][neurons[0]]= 1.0; // bias Layer[0] = 1
        NeuronCombination[1][neurons[1]]= 1.0; // bias Layer[1] = 1
        NeuronCombination[2][neurons[2]]= 1.0; // bias Layer[2] = 1 // (Recycle Bin) Not Used Now

        correct=0;
        InitialWeight();
    }

    void training()
    {
        for(k=0;k<NumberOfTrainData;++k)
        {
            //if(k%1000==0) printf("K :: %i\n",k); //debug

            for(l=0;l<Layers;++l) for(i=0;i<neurons[l];++i) NeuronCombination[l][i] = NeuronActivation[l][i] = NeuronError[l][i] = 0;

            // Calculate ( Combination Function , Activation Function ) For Neural Network
            for(i=0;i<Pixels;++i) NeuronCombination[0][i] = NeuronActivation[0][i] = TrainData[k][i] / 256.0 + 0.0001;
            for(l=1;l<Layers;++l)
            {
                for(j=0;j<neurons[l];++j) for (i = 0; i <= neurons[l - 1]; ++i) NeuronCombination[l][j] += (NeuronActivation[l - 1][i] * NeuronWeight[l][i][j]);
                for(j=0;j<neurons[l];++j) NeuronActivation[l][j]= 0.2 / (0.2 + exp(-NeuronCombination[l][j])); // decreasing sigmoid constant for decreasing saturating chance
            }

            // Calculate Errors For All Neurons In Neural Network (Output Layer Neurons --Then--> Hidden Layers Neurons)
            for(i=0;i<=neurons[Layers-1];++i)
            {
                if (i != TrainResult[k]) NeuronError[Layers - 1][i] = NeuronActivation[Layers - 1][i];
                else NeuronError[Layers - 1][i] = NeuronActivation[Layers - 1][i] - 1.0;
            }
            for(l=Layers-2;l>0;--l) for(i=0;i<neurons[l];++i) for(j=0;j<neurons[l+1];++j) NeuronError[l][i]+=(NeuronWeight[l+1][i][j]/SumWeights[l+1][j])*NeuronError[l+1][j];

            // Modify NeuronWeights For Neural Network
            for(l=Layers-1;l>0;--l) for(i=0;i<=neurons[l-1];++i) for(j=0;j<neurons[l];++j)
            {
                NewDelta = 2 * NeuronError[l][j] * NeuronActivation[l][j] * (1.0 - NeuronActivation[l][j]) * NeuronActivation[l-1][i];

                NeuronDeltaWeight[l][i][j] = NeuronDeltaWeight[l][i][j] * MomentumRate    +    NewDelta * LearningRate;

                SumWeights[l][j]-=abs(NeuronWeight[l][i][j]);
                NeuronWeight[l][i][j]-=NeuronDeltaWeight[l][i][j];
                SumWeights[l][j]+=abs(NeuronWeight[l][i][j]);
            }
        }
    }

    void testing()
    {
        for(k=0;k<NumberOfTestData;++k)
        {
            for(l=0;l<Layers;++l) for(i=0;i<neurons[l];++i) NeuronCombination[l][i] = NeuronActivation[l][i] = NeuronError[l][i]=0;

            // Calculate ( Combination Function , Activation Function ) For Neural Network
            for(i=0;i<Pixels;++i) NeuronCombination[0][i] = NeuronActivation[0][i] = TestData[k][i] / 256.0 + 0.0001;
            for(l=1;l<Layers;++l)
            {
                for(j=0;j<neurons[l];++j) for (i = 0; i <= neurons[l - 1]; ++i) NeuronCombination[l][j] += (NeuronActivation[l - 1][i] * NeuronWeight[l][i][j]);
                for(j=0;j<neurons[l];++j) NeuronActivation[l][j]= 0.2 / (0.2 + exp(-NeuronCombination[l][j])); // decreasing sigmoid constant for decreasing saturating chance
            }

            result=0;
            for(i=0;i<10;++i) if(NeuronActivation[Layers - 1][i] > NeuronActivation[Layers - 1][result]) result=i;
            if(result==TestResult[k]) ++correct;
        }
    }

private:
    void InitialWeight()
    {
        for(l=0;l<Layers;++l) for(i=0;i<Neurons;++i) SumWeights[l][i]=0;
        for(l=1;l<Layers;++l)
        {
            tmp= (double)(1 << 15) * pow(neurons[l-1],0.5); // max random value can generated by rand() function   *   number of neurons in previous layer
            for (i = 0; i <= neurons[l-1]; ++i) for (j = 0; j < neurons[l]; ++j)
            {
                k = rand();
                NeuronWeight[l][i][j]= k / tmp;

                SumWeights[l][j] += NeuronWeight[l][i][j];
                if (k % 3 == 0) NeuronWeight[l][i][j] = -NeuronWeight[l][i][j];
            }
        }
    }
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main()
{
    FILE *f;
    f=fopen("mnist_train.csv","r");
    fread(c,200000000,1,f);
    fclose(f);
    for(i=0,k=0;k<NumberOfTrainData;++k)
    {
        TrainResult[k]=c[i]-'0'; i+=2;
        for(j=0;j<Pixels;++j,++i) for(;c[i]>='0' && c[i]<='9';++i) TrainData[k][j]=TrainData[k][j]*10+c[i]-'0';
    }

    for(k=0;k<200000000;++k) c[k]='\0';
    f=fopen("mnist_test.csv","r");
    fread(c,200000000,1,f);
    fclose(f);
    for(i=0,k=0;k<NumberOfTestData;++k)
    {
        TestResult[k]=c[i]-'0'; i+=2;
        for(j=0;j<Pixels;++j,++i) for(;c[i]>='0' && c[i]<='9';++i) TestData[k][j]=TestData[k][j]*10+c[i]-'0';
    }

    int start,end;
    start=clock();
    MNIST_NN Agent;

    printf("Number Of Epochs == %i\n\n",Epoch);
    printf("Momentum Rate == %f\n",MomentumRate);
    printf("Scheduling Momentum Rate == %f\n\n",SchedulingMomentumRate);
    printf("Learning Rate == %f\n",LearningRate);
    printf("Scheduling Learning Rate == %f\n\n",SchedulingLearningRate);

    for(int e=1;e<=Epoch;++e) {printf("Epoch :: %i\n",e); Agent.training(); LearningRate*=SchedulingLearningRate;}
    Agent.testing();
    end=clock();

    printf("Correct == %i\n",correct);
    printf("Time == %i\n",end-start);

    return 0;
}
