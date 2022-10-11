# ANPR
Automatic number-plate recognition


Passos per executar:
  - Carpeta "images" en el directori des d'on s'executa el programa amb les imatges que es vol reconèixer. El nom dels arxius ha de contenir la matrícula si es vol poder avaluar els resultats en format sense espais (ex: 7020DMR.jpg)
  - Els resultats es guardaràn en les següents carpetes en el mateix directori d'execució
    - "licensePlate/incorrect": localitzacions (correcte o incorrecte) de matrícules amb reconeixement erroni
    - "licensePlate/incorrect": localitzacions de matrícules amb reconeixement correcte
    - "licensePlate/boundingBox": imatges originals amb la bounding box de la matrícula detectada i el reconeixement de caràcters
    
Paràmetres:
  - "--debug" : si està especificat, mostra per pantalla les imatges resultants d'aplicar les transformacions durant la localització


Projecte realitzar per:
  - Pol Fombona Delgado
  - Victor Garcia Pina
  - Ángel Funes Olaria
  - Joel Soler Huix
