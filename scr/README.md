# Modifiche fatte rispetto al paper:



1. Nel paper il è CNNblock costituito d LeakyReLU -> BN -> CNN. Ora invece è *CNN -> BN -> SiLU*
3. Modifica di (CNNBlock3 + CNNBlock4): ora l'output finale è B x 512 x 3 x 6  
