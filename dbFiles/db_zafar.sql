CREATE TABLE Message (
  idMessage SERIAL  NOT NULL ,
  time TIMESTAMP    ,
  name VARCHAR(255)    ,
  message TEXT    ,
  email VARCHAR(255)      ,
PRIMARY KEY(idMessage));



CREATE TABLE Keywords (
  idKeywords SERIAL  NOT NULL ,
  key1 VARCHAR(45)    ,
  key2 VARCHAR(45)    ,
  key3 VARCHAR(45)    ,
  key4 VARCHAR(45)    ,
  key5 VARCHAR(45)    ,
  time TIMESTAMP      ,
PRIMARY KEY(idKeywords));



CREATE TABLE Paragraph (
  idParagraph SERIAL  NOT NULL ,
  Keywords_idKeywords INTEGER   NOT NULL ,
  para TEXT      ,
PRIMARY KEY(idParagraph)  ,
  FOREIGN KEY(Keywords_idKeywords)
    REFERENCES Keywords(idKeywords));


CREATE INDEX Paragraph_FKIndex1 ON Paragraph (Keywords_idKeywords);



CREATE TABLE Feedback (
  idFeedback SERIAL  NOT NULL ,
  Paragraph_idParagraph INTEGER   NOT NULL ,
  coherency INTEGER    ,
  relevance INTEGER    ,
  grammar INTEGER      ,
PRIMARY KEY(idFeedback)  ,
  FOREIGN KEY(Paragraph_idParagraph)
    REFERENCES Paragraph(idParagraph));


CREATE INDEX Feedback_FKIndex1 ON Feedback (Paragraph_idParagraph);




