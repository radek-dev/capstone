/*
SHOW DATABASES;
CREATE DATABASE capstone;
SHOW TABLES;
*/

SHOW ENGINE INNODB STATUS

create or replace table symbolRef
(
  id      int unsigned not null,
  symbol  varchar(20) not null,
  constraint symbol_pk PRIMARY KEY (symbol)
) ENGINE = InnoDB;
# insert some reference data
insert into symbolRef values (1, 'TUSDBTC');

create or replace table bid
(
  price     decimal(20, 10)  not null,
  amount    bigint           null,
  updatedId bigint           not null,
  myUtc     datetime         null,
  symbolId  int unsigned     not null,
  constraint bid_pk PRIMARY KEY (price, updatedId, symbolId),
  constraint `fk_bid_symbolRef` FOREIGN KEY (symbolId) REFERENCES symbolRef (id)
    	ON DELETE CASCADE
		  ON UPDATE RESTRICT
) ENGINE = InnoDB;




create or replace table ask
(
  `index`   bigint   null,
  price     double   null,
  amount    double   null,
  updatedId bigint   null,
  myUtc     datetime null,
  symbol    text     null
);




