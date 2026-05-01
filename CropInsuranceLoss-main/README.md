# CropInsuranceLoss
for downloading, reading, and extracting cause-of-loss data from USDA RMA records

## Getting Started

### Dependencies

Python Libraries:

* requests
* zipfile
* os
* io
* pandas
* numpy

### Executing program

* download zipped crop insurance data from www.rma.usda.gov by running
```
python -W ignore read_crop_loss_api.py
```
* this will download crop insurance loss data and unzip the files into 
* a subdirectory called 'crop_loss_data', with three sub-sub-directories: 'state_county_crop', 'type_practice_usage', 'cost_of_loss'
* create csv files with cause-of-loss, insured loss, insured value, and insured acreage data:
```
python -W ignore spark_analysis.py
```
* this will create loss data files for each individual state, as well as a nationwide loss data files
* new data is created in subdirectory 'crop_loss_data/readable_col_data_by_state'

## Data Fields

The following table describes the fields in the crop insurance data:
Crop Insurance Experience Coverage Level Type Practice Unit Structure: this is for SOBTPU

| Field | Format | Definition |
|-------|--------|------------|
| 1* Commodity Year | 9(04) | The year in which the crop is normally harvested and the year for which coverage is provided. |
| 2* State Code | X(02) | The FIPS/ANSI code of the State in which the insured farm is located. |
| 3 State Name | X(30) | The name of the State in which the insured farm is located. |
| 4 State Abbreviation | X(02) | The postal abbreviation of the State in which the insured farm is located. |
| 5* County Code | X(03) | The FIPS/ANSI code of the County in which the crop is located. |
| 6 County Name | X(40) | The name of the County in which the insured farm is located. |
| 7* Commodity Code | X(04) | The RMA code of the commodity for which the policy is issued. |
| 8 Commodity Name | X(40) | The name of the commodity for which the policy is issued. |
| 9* Insurance Plan Code | X(02) | The RMA code of the type of insurance plan for which the policy is issued. |
| 10 Insurance Plan Abbreviation | X(10) | The abbreviation of the name of the type of insurance plan for which the policy is issued. |
| 11* Coverage Type Code | X(01) | The type of coverage: CAT vs. Buy-up. |
| 12* Coverage Level Percent | 9(01)V9(04) | The level of coverage selected for the policy insured, similar to a deductible. Expressed as a decimal. |
| 13* Delivery ID | X(01) | The delivery type: Reinsured policies vs. policies administered through FCIC or FSA. |
| 14* Type Code | X(03) | The RMA code of the commodity type for which the policy is issued denoting a type, class or variety. |
| 15 Type Name | X(50) | The name of the commodity type for which the policy is issued. |
| 16* Practice Code | X(03) | The RMA code of the practice for which the policy is issued denoting a specific farming practice used by the producer (e.g. irrigated, non- irrigated, no practice specified). |
| 17 Practice Name | X(50) | The name of the practice for which the policy is issued. |
| 18* Unit Structure Code | X(02) | The RMA code of the unit structure for which the policy is issued. |
| 19 Unit Structure Name | X(50) | The name of the unit structure for which the policy is issued. |
| 20 Net Reporting Level Amount | 9(16) | The quantity of the commodity insured adjusted by the insured's share of the commodity. Depending on the commodity, this could be acres, tons, colonies, pounds, or trees as identified by the Commodity Reporting Level Type. |
| 21* Reporting Level Type | X(25) | The description of the quantity type associated with Net Commodity Reporting Level Amount. Values include acres, tons, colonies, pounds, or trees. |
| 22 Liability Amount | 9(13) | The maximum amount of insurance in U.S. Dollars. |
| 23 Total Premium Amount | 9(13) | The premium before application of any subsidies in U.S. Dollars. |
| 24 Subsidy Amount | 9(13) | The amount of subsidized premium in U.S. Dollars. |
| 25 Indemnity Amount | 9(13) | The amount of the loss in U.S. Dollars. |
| 26 Loss Ratio | 9(06)V9(02) | Ratio of losses to premium (indemnity / total premium) |
| 27 Endorsed Commodity Reporting Level Amount | 9(16) | Number of acres insured under an endorsement (e.g. SCO, STAX, Margin Protection)
