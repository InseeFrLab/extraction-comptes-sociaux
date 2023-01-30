################################################################################'
# Fonction principale de récupération des comptes sociaux
################################################################################'
recupererComptes <- function(siren,annee,base_dossier = paste0(tempdir(),'/',as.character(siren))){
  connexionApi()
  debut_date_depot <- paste0(annee,"0101")
  fin_date_depot <- paste0(annee, "1231")
  retour <- httr::GET(requete_selection_idfichier(siren,debut_date_depot,fin_date_depot), config = httr::add_headers(login = "briant", password = "Dimanche2021" ))
  donnees_retournees <- jsonlite::fromJSON(httr::content(retour,as="text"))
  id_fichier <- donnees_retournees$idFichier[length(donnees_retournees$idFichier)]
  
  if(is.null(id_fichier)){
    print(paste("Pas de données pour l'UL: ",siren))
  }
  else{
    retour2 <- httr::GET(requete_fichier(id_fichier))
    ecrire_pdf_retour(retour2,siren,base_dossier = base_dossier)
  }
  deconnexionAPI()
}

################################################################################'
# Connexion/ Déconnexion de l'API
################################################################################'

#Fonction de connexion à l'API RNCS
connexionApi <- function(){
  # Configuration proxy Insee
  httr::set_config(use_proxy("http://proxy-rie.http.insee.fr", 8080))
  #reset_config()
  #URL de l'API RNCS
  query_entree <- paste0("https://opendata-rncs.inpi.fr/services/diffusion/login")
  # Requ?te POST de connexion à l'API
  entree_inpi <- httr::POST(url = query_entree, config = httr::add_headers(login = "briant", password = "Drisse240" ))
  return(entree_inpi$status_code)
}

# Fonction de déconnexion à l'API RNCS
deconnexionAPI <- function(){
  query_sortie <- "https://opendata-rncs.inpi.fr/services/diffusion/logout"
  sortie_inpi <- httr::POST(url = query_sortie)
  return(sortie_inpi$status_code)
}

################################################################################'
# Consolidation d'une EP
################################################################################'

# Fonction de récupération des comptes sociaux
ecrire_pdf_retour <- function(retour_GET,siren,base_dossier = paste0(tempdir(),'/',as.character(siren),'/')){
  ## création des diff?rents r?pertoires s'ils n'existent pas
  if(!dir.exists(base_dossier))dir.create(base_dossier)
  
  dosstemp <- base_dossier
  writeBin(retour_GET$content,paste0(base_dossier,as.character(siren),".zip"))

  ## Dézippe le fichier téléchargé
  unzip(paste0(dosstemp,as.character(siren),".zip"),exdir = substr(base_dossier,1,nchar(base_dossier)-1) )
  file.remove(paste0(dosstemp,as.character(siren),".zip"))
  #########################
  ## Ecriture des fichiers pdf et xml contenant les comptes annuels et les métadonnées
  
  fichier_pdf<- list.files(dosstemp,pattern="zip",full.names = T)
  if (length(fichier_pdf)>0){
    unzip(fichier_pdf,exdir = dosstemp)
  }else{
    unlink(dosstemp,recursive = TRUE)
  }
  fichiers_zip <- list.files(dosstemp,pattern="zip",full.names = T)
  fichier_md5 <- list.files(dosstemp,pattern="md5",full.names = T)
  fichier_json <- list.files(dosstemp,pattern="json",full.names = T)
  
  ## Supprime les répertoires de travail
  file.remove(fichiers_zip)
  file.remove(fichier_md5)
  file.remove(fichier_json)
}

# Fonction de construction de la requête pour l'API pour obtenir l'id du fichier voulu
requete_selection_idfichier <- function(siren, dateDebutDepot, dateFinDepot){
  requete <- paste0("https://opendata-rncs.inpi.fr/services/diffusion/bilans/find?siren=",siren,"&dateDepotDebut=",dateDebutDepot,"&dateDepotFin=",dateFinDepot)
  return(requete)
}

# Fonction de construction de la requête pour l'API pour obtenir les comptes sociaux
requete_fichier <- function(idFichier){
  return(paste0("https://opendata-rncs.inpi.fr/services/diffusion/document/get?listeIdFichier=",idFichier))
}
