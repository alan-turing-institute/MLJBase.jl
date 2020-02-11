# Data API
# The structures are based on these descriptions
# https://github.com/openml/OpenML/tree/master/openml_OS/views/pages/api_new/v1/xsd
# https://www.openml.org/api_docs#!/data/get_data_id

# To do:
# - Save the file in a local folder
# - Check downloaded files in local folder before downloading it again
# - Use local stored file whenever possible

api_url = "https://www.openml.org/api/v1/json"

"""
Returns information about a dataset. The information includes the name,
information about the creator, URL to download it and more.

110 - Please provide data_id.
111 - Unknown dataset. Data set description with data_id was not found in the database.
112 - No access granted. This dataset is not shared with you.
"""
function getDatasetDescription(id::Int; api_key::String="")
    url = string(api_url, "/data/$id")
    try
        r = HTTP.request("GET", url)
        if r.status == 200
            return JSON.parse(String(r.body))
        elseif r.status == 110
            println("Please provide data_id.")
        elseif r.status == 111
            println("Unknown dataset. Data set description with data_id was not found in the database.")
        elseif r.status == 112
            println("No access granted. This dataset is not shared with you.")
        end
    catch e
        return "Error occurred : $e"
    end
end

"""
Returns a DataFrame
"""
function getDataframeFromOpenmlAPI(id::Int)
    response = getDatasetDescription(id)
    arff_file = HTTP.request("GET", response["data_set_description"]["url"])
    df = convertArffToDataFrame(arff_file)
    return df
end

"""
Returns a DataFrame from the HTTP.response requested to the OpenML API.
"""
function convertArffToDataFrame(response)
    ##possible types in arff files are:
    #real, numeric,
    #date
    #string
    #nominal specifications, starting with {

    data = String(response.body)
    data2 = split(data, "\n")

    featureNames = String[]
    dataTypes = String[]
    dataset = []

    for line in data2
        if length(line) > 0
            if line[1:1] != "%"
                d = []
                if occursin("@attribute", lowercase(line))
                    push!(featureNames, replace(split(line, " ")[2], "'" => ""))
                    push!(dataTypes, split(line, " ")[3])
                elseif occursin("@relation", lowercase(line))
                    nothing
                elseif occursin("@data", lowercase(line))
                    # it means the data starts so we can create the data frame
                    nothing
                else
                    values = split(line, ",")
                    for i = 1:length(featureNames)
                        if in(lowercase(dataTypes[i]), ["real","numeric"])
                            push!(d, featureNames[i] => parse(Float64, values[i]))
                        else
                            # all the rest will be considered as String
                            push!(d, featureNames[i] => values[i])
                        end
                    end
                    push!(dataset, Dict(d))
                    #d = Dict(featureNames[i] => values[i] for i = 1:length(featureNames))

                end
            end
        end
    end

    df = DataFrame()
    namelist  = Symbol.(featureNames)
    for (i, name) in enumerate(namelist)
        df[name] =  [dataset[j][String(name)] for j in 1:length(dataset)]
    end

    return df
end

"""
Returns a list of all data qualities in the system.

412 - Precondition failed. An error code and message are returned
370 - No data qualities available. There are no data qualities in the system.
"""
function getDataQualitiesList()
    url = string(api_url, "/data/qualities/list")
    try
        r = HTTP.request("GET", url)
        if r.status == 200
            return JSON.parse(String(r.body))
        elseif r.status == 370
            println("No data qualities available. There are no data qualities in the system.")
        end
    catch e
        return "Error occurred : $e"
    end
end

"""
Returns a list of all data qualities in the system.

271 - Unknown dataset. Data set with the given data ID was not found (or is not shared with you).
272 - No features found. The dataset did not contain any features, or we could not extract them.
273 - Dataset not processed yet. The dataset was not processed yet, features are not yet available. Please wait for a few minutes.
274 - Dataset processed with error. The feature extractor has run into an error while processing the dataset. Please check whether it is a valid supported file. If so, please contact the API admins.
"""
function getDataFeatures(id::Int; api_key::String = "")
    if api_key == ""
        url = string(api_url, "/data/features/$id")
    end
    r = HTTP.request("GET", url)
    if r.status == 200
        return JSON.parse(String(r.body))
    elseif r.status == 271
        println("Unknown dataset. Data set with the given data ID was not found (or is not shared with you).")
    elseif r.status == 272
        println("No features found. The dataset did not contain any features, or we could not extract them.")
    elseif r.status == 273
        println("Dataset not processed yet. The dataset was not processed yet, features are not yet available. Please wait for a few minutes.")
    elseif r.status == 274
        println("Dataset processed with error. The feature extractor has run into an error while processing the dataset. Please check whether it is a valid supported file. If so, please contact the API admins.")
    end

end

"""
Returns the qualities of a dataset.

360 - Please provide data set ID
361 - Unknown dataset. The data set with the given ID was not found in the database, or is not shared with you.
362 - No qualities found. The registered dataset did not contain any calculated qualities.
363 - Dataset not processed yet. The dataset was not processed yet, no qualities are available. Please wait for a few minutes.
364 - Dataset processed with error. The quality calculator has run into an error while processing the dataset. Please check whether it is a valid supported file. If so, contact the support team.
365 - Interval start or end illegal. There was a problem with the interval start or end.
"""
function getDataQualities(id::Int; api_key::String = "")
    if api_key == ""
        url = string(api_url, "/data/qualities/$id")
    end
    r = HTTP.request("GET", url)
    if r.status == 200
        return JSON.parse(String(r.body))
    elseif r.status == 360
        println("Please provide data set ID")
    elseif r.status == 361
        println("Unknown dataset. The data set with the given ID was not found in the database, or is not shared with you.")
    elseif r.status == 362
        println("No qualities found. The registered dataset did not contain any calculated qualities.")
    elseif r.status == 363
        println("Dataset not processed yet. The dataset was not processed yet, no qualities are available. Please wait for a few minutes.")
    elseif r.status == 364
        println("Dataset processed with error. The quality calculator has run into an error while processing the dataset. Please check whether it is a valid supported file. If so, contact the support team.")
    elseif r.status == 365
        println("Interval start or end illegal. There was a problem with the interval start or end.")
    end
end

"""
List datasets, possibly filtered by a range of properties.
Any number of properties can be combined by listing them one after
the other in the
form '/data/list/{filter}/{value}/{filter}/{value}/...'
Returns an array with all datasets that match the constraints.

Any combination of these filters /limit/{limit}/offset/{offset} -
returns only {limit} results starting from result number {offset}.
Useful for paginating results. With /limit/5/offset/10,
    results 11..15 will be returned.

Both limit and offset need to be specified.
/status/{status} - returns only datasets with a given status,
either 'active', 'deactivated', or 'in_preparation'.
/tag/{tag} - returns only datasets tagged with the given tag.
/{data_quality}/{range} - returns only tasks for which the
underlying datasets have certain qualities.
{data_quality} can be data_id, data_name, data_version, number_instances,
number_features, number_classes, number_missing_values. {range} can be a
specific value or a range in the form 'low..high'.
Multiple qualities can be combined, as in
'number_instances/0..50/number_features/0..10'.

370 - Illegal filter specified.
371 - Filter values/ranges not properly specified.
372 - No results. There where no matches for the given constraints.
373 - Can not specify an offset without a limit.
"""
function getListAndFilter(filters::String; api_key::String = "")
    if api_key == ""
        url = string(api_url, "/data/list/$filters")
    end
    r = HTTP.request("GET", url)
    if r.status == 200
        return JSON.parse(String(r.body))
    elseif r.status == 370
        println("Illegal filter specified.")
    elseif r.status == 371
        println("Filter values/ranges not properly specified.")
    elseif r.status == 372
        println("No results. There where no matches for the given constraints.")
    elseif r.status == 373
        println("Can not specify an offset without a limit.")
    end
end

"""
This call is for people running their own dataset processing engines.
It returns the details of datasets that are not yet processed by
the given processing engine. It doesn't process the datasets,
it just returns the dataset info.

data_engine_id : The ID of the data processing engine. You get this ID when you register
a new data processing engine with OpenML. The ID of the main
data processing engine is 1.

order : When there are multiple datasets still to process,
this defines which ones to return. Options are 'normal' - the oldest
datasets, or 'random'.

412 - Precondition failed. An error code and message are returned.
681 - No unprocessed datasets.
"""
function getListUnprocessedDatasets(data_engine_id::String, order::String, api_key::String = "")
    if api_key == ""
        url = string(api_url, "/data/unprocessed/$data_engine_id/$order")
    end
    r = HTTP.request("GET", url)
    if r.status == 200
        return JSON.parse(String(r.body))
    elseif r.status == 681
        println("No unprocessed datasets.")
    end
end

"""
Uploads a dataset. Upon success, it returns the data id.

description : An XML file describing the dataset. Only name, description,
and data format are required. Also see the XSD schema and an XML example.

dataset : The actual dataset, being an ARFF file.

api_key : Api key to authenticate the user

Precondition failed. An error code and message are returned.
130 - Problem with file uploading. There was a problem with the file upload.
131 - Problem validating uploaded description file. The XML description format does not meet the standards.
132 - Failed to move the files. Internal server error, please contact API administrators.
133 - Failed to make checksum of datafile. Internal server error, please contact API administrators.
134 - Failed to insert record in database. Internal server error, please contact API administrators.
135 - Please provide description xml.
136 - File failed format verification. The uploaded file is not valid according to the selected file format. Please check the file format specification and try again.
137 - Please provide API key. In order to share content, please log in or provide your API key.
138 - Authentication failed. The API key was not valid. Please try to login again, or contact API administrators
139 - Combination name / version already exists. Leave version out for auto increment
140 - Both dataset file and dataset url provided. The system is confused since both a dataset file (post) and a dataset url (xml) are provided. Please remove one.
141 - Neither dataset file or dataset url are provided. Please provide either a dataset file as POST variable, or a dataset url in the description XML.
142 - Error in processing arff file. Can be a syntax error, or the specified target feature does not exists. For now, we only check on arff files. If a dataset is claimed to be in such a format, and it can not be parsed, this error is returned.
143 - Suggested target feature not legal. It is possible to suggest a default target feature (for predictive tasks). However, it should be provided in the data.
144 - Unable to update dataset. The dataset with id could not be found in the database. If you upload a new dataset, unset the id.
"""
function uploadDataset(description, dataset, api_key::String)
    url = string(api_url, "/data")
    r = HTTP.request("POST", url)
    if r.status == 200
        return JSON.parse(String(r.body))
    elseif r.status == 130
        println("Problem with file uploading. There was a problem with the file upload.")
    elseif r.status == 131
        println("Problem validating uploaded description file. The XML description format does not meet the standards.")
    elseif r.status == 132
        println("Failed to move the files. Internal server error, please contact API administrators.")
    elseif r.status == 133
        println("Failed to make checksum of datafile. Internal server error, please contact API administrators.")
    elseif r.status == 134
        println("Failed to insert record in database. Internal server error, please contact API administrators.")
    elseif r.status == 135
        println("Please provide description xml.")
    elseif r.status == 136
        println("File failed format verification. The uploaded file is not valid according to the selected file format. Please check the file format specification and try again.")
    elseif r.status == 137
        println("Please provide API key. In order to share content, please log in or provide your API key.")
    elseif r.status == 138
        println("Authentication failed. The API key was not valid. Please try to login again, or contact API administrators")
    elseif r.status == 139
        println("Combination name / version already exists. Leave version out for auto increment")
    elseif r.status == 140
        println("Both dataset file and dataset url provided. The system is confused since both a dataset file (post) and a dataset url (xml) are provided. Please remove one.")
    elseif r.status == 141
        println("Neither dataset file or dataset url are provided. Please provide either a dataset file as POST variable, or a dataset url in the description XML.")
    elseif r.status == 142
        println("Error in processing arff file. Can be a syntax error, or the specified target feature does not exists. For now, we only check on arff files. If a dataset is claimed to be in such a format, and it can not be parsed, this error is returned.")
    elseif r.status == 143
        println("Suggested target feature not legal. It is possible to suggest a default target feature (for predictive tasks). However, it should be provided in the data.")
    elseif r.status == 144
        println("Unable to update dataset. The dataset with id could not be found in the database. If you upload a new dataset, unset the id.")
    end
end

"""
Uploads dataset qualities. Upon success, it returns the data id.

Parameters

description : An XML file describing the dataset. Only name, description,
and data format are required. Also see the XSD schema and an XML example.

api_key : Api key to authenticate the user

381 - Something wrong with XML, please check did and evaluation_engine_id
382 - Please provide description xml
383 - Problem validating uploaded description file
384 - Dataset not processed yet

"""
function uploadDatasetQualities(description::String, api_key::String)
    url = string(api_url, "/data/qualities")
    r = HTTP.request("POST", url)
    if r.status == 200
        return JSON.parse(String(r.body))
    elseif r.status == 381
        println("Something wrong with XML, please check did and evaluation_engine_id")
    elseif r.status == 382
        println("Please provide description xml")
    elseif r.status == 383
        println("Problem validating uploaded description file")
    elseif r.status == 384
        println("Dataset not processed yet")
    end
end

"""
Change the status of a dataset, either 'active' or 'deactivated'

691 - Illegal status
692 - Dataset does not exists
693 - Dataset is not owned by you
694 - Illegal status transition
695 - Status update failed
"""
function changeStatusDataset(id::String, status::String, api_key::String)
    url = string(api_url, "/data/status/update")
    r = HTTP.request("POST", url)
    if r.status == 200
        return JSON.parse(String(r.body))
    elseif r.status == 691
        println("Illegal status")
    elseif r.status == 692
        println("Dataset does not exists")
    elseif r.status == 693
        println("Dataset is not owned by you")
    elseif r.status == 694
        println("Illegal status transition")
    elseif r.status == 695
        println("Status update failed")
    end
end

"""
Uploads dataset feature description. Upon success, it returns the data id.

412 - Precondition failed. An error code and message are returned.
431 - Dataset already processed
432 - Please provide description xml
433 - Problem validating uploaded description file
434 - Could not find dataset
436 - Something wrong with XML, check data id and evaluation engine id
"""
function updaloadDatasetFeatureDescription(description, api_key::String)
    url = string(api_url, "/data/features")
    r = HTTP.request("POST", url)
    if r.status == 200
        return JSON.parse(String(r.body))
    end
end

"""
Tags a dataset.

412 - Precondition failed. An error code and message are returned.
470 - In order to add a tag, please upload the entity id (either data_id, flow_id, run_id) and tag (the name of the tag).
471 - Entity not found. The provided entity_id {data_id, flow_id, run_id} does not correspond to an existing entity.
472 - Entity already tagged by this tag. The entity {dataset, flow, run} already had this tag.
473 - Something went wrong inserting the tag. Please contact OpenML Team.
474 - Internal error tagging the entity. Please contact OpenML Team.
"""
function tagDatabase(data_id::Int, tag::String, api_key::String)
    url = string(api_url, "/data/tag")
    r = HTTP.request("POST", url)
    if r.status == 200
        return JSON.parse(String(r.body))
    end
end

"""
Untags a dataset.

412 - Precondition failed. An error code and message are returned.
475 - Please give entity_id {data_id, flow_id, run_id} and tag. In order
to remove a tag, please upload the entity id (either data_id, flow_id, run_id)
and tag (the name of the tag).
476 - Entity {dataset, flow, run} not found. The provided entity_id
{data_id, flow_id, run_id} does not correspond to an existing entity.
477 - Tag not found. The provided tag is not associated with the
entity {dataset, flow, run}.
478 - Tag is not owned by you. The entity {dataset, flow, run} was
tagged by another user. Hence you cannot delete it.
479 - Internal error removing the tag. Please contact OpenML Team.
"""
function untagDatabase(data_id::Int, tag::String, api_key::String)
    url = string(api_url, "/data/untag")
    r = HTTP.request("POST", url)
    if r.status == 200
        return JSON.parse(String(r.body))
    end
end

"""
412 - Precondition failed. An error code and message are returned.
686 - Please specify the features the evaluation engine wants to calculate
(at least 2).
687 - No unprocessed datasets according to the given set of meta-features.
688 - Illegal qualities.
"""
function getListOfDatasetsWithUnprocessedQualities(data_engine_id::String, order::String; api_key::String, qualities::String)
    url = string(api_url, "/data/qualities/unprocessed/{$data_engine_id}/{$order}")
    r = HTTP.request("POST", url)
    if r.status == 200
        return JSON.parse(String(r.body))
    end
end

"""
Deletes a dataset. Upon success, it returns the ID of the deleted dataset.
"""
function deleteDataset(id::Int; api_key::String)
    url = string(api_url, "/data/{$id}")
    r = HTTP.request("DELETE", url)
    if r.status == 200
        return JSON.parse(String(r.body))
    end
end

# Flow API

# Task API

# Run API