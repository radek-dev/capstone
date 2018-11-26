/*
SHOW DATABASES;
CREATE DATABASE capstone;
SHOW TABLES;
*/

# use this for error reporting
SHOW ENGINE INNODB STATUS;

create or replace table symbolRef
(
  id      int unsigned not null,
  symbol  varchar(20) not null,
  constraint symbol_pk PRIMARY KEY (symbol)
) ENGINE = InnoDB;

create index ix_symbol_index
  on symbolRef (`id`);
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
    	ON DELETE RESTRICT
		  ON UPDATE RESTRICT
) ENGINE = InnoDB;


create or replace table ask
(
  price     decimal(20, 10)  not null,
  amount    bigint           null,
  updatedId bigint           not null,
  myUtc     datetime         null,
  symbolId  int unsigned     not null,
  constraint ask_pk PRIMARY KEY (price, updatedId, symbolId),
  constraint `fk_ask_symbolRef` FOREIGN KEY (symbolId) REFERENCES symbolRef (id)
    	ON DELETE RESTRICT
		  ON UPDATE RESTRICT
) ENGINE = InnoDB;


create table limits
(
  `index`       bigint null,
  `interval`    text   null,
  intervalNum   bigint null,
  `limit`       bigint null,
  rateLimitType text   null
);

create index ix_limits_index
  on limits (`index`);

create table data_dict
(
  data_field varchar(128) not null
    primary key,
  definition varchar(300) null
)
  comment 'store the data dictionary for columns within the database';

# import some data directly
INSERT INTO capstone.data_dict (data_field, definition) VALUES ('amount', 'quoted amount to trade for the quoted price');
INSERT INTO capstone.data_dict (data_field, definition) VALUES ('price', 'quoted bid or ask price based on the table name');
INSERT INTO capstone.data_dict (data_field, definition) VALUES ('symbol', 'currency pair ID');
INSERT INTO capstone.data_dict (data_field, definition) VALUES ('updatedId', 'Time stamp from the exchange for the given set quotes');
INSERT INTO capstone.data_dict (data_field, definition) VALUES ('UTC time stamp from the computer where the code run', 'UTC time stamp from the computer where the code run');