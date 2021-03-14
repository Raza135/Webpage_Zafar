CREATE TABLE Message (
  idMessage INTEGER UNSIGNED  NOT NULL   AUTO_INCREMENT,
  time TIMESTAMP  NULL  ,
  name VARCHAR(255)  NULL  ,
  message TEXT  NULL  ,
  email VARCHAR(255)  NULL    ,
PRIMARY KEY(idMessage));



CREATE TABLE Keywords (
  idKeywords INTEGER UNSIGNED  NOT NULL   AUTO_INCREMENT,
  key1 VARCHAR(45)  NULL  ,
  key2 VARCHAR(45)  NULL  ,
  key3 VARCHAR(45)  NULL  ,
  key4 VARCHAR(45)  NULL  ,
  key5 VARCHAR(45)  NULL  ,
  time TIMESTAMP  NULL    ,
PRIMARY KEY(idKeywords));



CREATE TABLE Paragraph (
  idParagraph INTEGER UNSIGNED  NOT NULL   AUTO_INCREMENT,
  Keywords_idKeywords INTEGER UNSIGNED  NOT NULL  ,
  para TEXT  NULL    ,
PRIMARY KEY(idParagraph)  ,
INDEX Paragraph_FKIndex1(Keywords_idKeywords),
  FOREIGN KEY(Keywords_idKeywords)
    REFERENCES Keywords(idKeywords)
      ON DELETE NO ACTION
      ON UPDATE NO ACTION);



CREATE TABLE Feedback (
  idFeedback INTEGER UNSIGNED  NOT NULL   AUTO_INCREMENT,
  Paragraph_idParagraph INTEGER UNSIGNED  NOT NULL  ,
  coherency INTEGER UNSIGNED  NULL  ,
  relevance INTEGER UNSIGNED  NULL  ,
  grammar INTEGER UNSIGNED  NULL    ,
PRIMARY KEY(idFeedback)  ,
INDEX Feedback_FKIndex1(Paragraph_idParagraph),
  FOREIGN KEY(Paragraph_idParagraph)
    REFERENCES Paragraph(idParagraph)
      ON DELETE NO ACTION
      ON UPDATE NO ACTION);




