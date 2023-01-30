rm(list=ls())

library(httr)
library(readr)
library(pdftools)
source("functions_API.R")

cheminpdfsplit <- 'C:/Users/RQSCMG/Documents/GitHub/importcsfile/data_pdfsplit/'
cheminpng <- 'C:/Users/RQSCMG/Documents/GitHub/importcsfile/data_png/'

# 1. Importation depuis l'API ---------------------------------------------

ul_ref_EPcible1_2020 <- read_delim("ul_ref_EPcible1_2020.csv", 
                                   delim = ",", escape_double = FALSE, trim_ws = TRUE)

for (i in 1:length(ul_ref_EPcible1_2020$ul_ref)){
  recupererComptes(ul_ref_EPcible1_2020$ul_ref[i], 2020)
}


# 2. Découpage en pdf d'une page contenant des tableaux -------------------


# CSV donnant les plages de pages où se trouvent les tableaux
ul_ref_pagestableau <- read_delim("ul_ref_pagestableau.csv", 
                                  delim = ",", escape_double = FALSE, trim_ws = TRUE)



# Première extraction le 17/02 jusqu'au 21ème élément car déjà beaucoup de pages pdf à traiter, annoter
for (j in 1:21){
  deb_j <- ul_ref_pagestableau[j,]$debut_tableau
  fin_j <- ul_ref_pagestableau[j,]$fin_tableau
  deb2_j <- ul_ref_pagestableau[j,]$debut_tableau2
  fin2_j <- ul_ref_pagestableau[j,]$fin_tableau2
  ul_j <- ul_ref_pagestableau$ul_ref[j]
  
  if (!is.na(deb_j)) {
    for (page_i in deb_j:fin_j) {
      pdf_subset(input = paste0('data_API/',ul_j,".pdf"), 
                 pages = page_i, output = paste0(cheminpdfsplit, ul_j, "_", as.character(page_i), ".pdf"))  
    }  
  }
  if (!is.na(deb2_j)) {
    for (page2_i in deb2_j:fin2_j) {
      pdf_subset(input = paste0('data_API/',ul_j,".pdf"), 
                 pages = page2_i, output = paste0(cheminpdfsplit, ul_j, "_", as.character(page2_i), ".pdf"))  
    }  
  }
}


# 3. Transformation des pdf en png, le passage en bmp sera fait su --------

#La transformation en bmp sera faite via le programme Python du stagiaire Adem : pngtobmp

#pdf_convert('dataimporte/subset.pdf',dpi = 300, format = 'png')


fichier_pdf<- list.files(cheminpdfsplit,pattern="pdf",full.names = T)

setwd(cheminpng)
for (fichier_i in fichier_pdf){
  pdf_convert(fichier_i,dpi = 300, format = 'png')
}

#test = pdf_render_page(fichier_pdf[1], page = 1, dpi = 300)

