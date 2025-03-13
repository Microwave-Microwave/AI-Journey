# This is a Showcase of My AI Projects

The projects are listed in reverse order of when I worked on them, starting with the most recent.

## Makemore
This is my most ambitious AI project yet.
"4_makemore/name_generator.py": showcases my ability to write well-documented code (I might have gone a little overboard with it 😅).

Makemore is language modeling based on the 2003 paper by Bengio. I modified the code so it is scalable; use any parameter and it will work. The context window is 5 by default. I also added automatic logging of training data (a graph) and a properties.txt file that saves any parameters used by the program so one can replicate previous findings.

I intend to use this as a template for future projects.

![Graph picture](https://i.imgur.com/t3QFCYA.png)

**Sample of generated names:** Jacie, Liviandela, Suina, Elia, Rosapelyn, Yanalyzah, Jawa, Aerin, Ivaakeria, Davean, Zayly, Marie, Jeznique, Cish, Deverody, Kilion, Nital, Callo, Zo, Cameaun

## Building Micrograd
Further experimentation with Micrograd, including the approximation of the values of the sinus function. Optimizations to the code were also implemented.

![Language structure](https://i.imgur.com/PsjvRFp.png)
This is a representation of what the model thinks about the relationship of letters when present in a name.

## The Spelled-out Intro to Language Modeling
This is Andrej Karpathy's tutorial series on language modeling. In the first lecture, we created a language bigram model and later approximated a probability table to predict the upcoming character of a name with a context window of 1.

I modified the Neural Network to train on the values of the sinus function, here you can see the learning process.

![Sinus wave approximation](https://i.imgur.com/oBHLhlb.gif)

## Harwards-CS50-AI
I started this Harvard course but later realized that it covers very basic knowledge. I completed the first two lectures.
